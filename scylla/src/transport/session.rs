use futures::{future::RemoteHandle, FutureExt};
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt::Display;
use std::net::SocketAddr;
use std::sync::{Arc, RwLock};
use tokio::net::{lookup_host, ToSocketAddrs};

use super::transport_errors::ConnectionError;
use crate::batch::Batch;
use crate::cql_to_rust::FromRow;
use crate::frame::response::cql_to_rust::FromRowError;
use crate::frame::response::result;
use crate::frame::response::Response;
use crate::frame::value::Value;
use crate::prepared_statement::PreparedStatement;
use crate::query::Query;
use crate::routing::{murmur3_token, Node, Shard, ShardInfo, Token};
use crate::transport::connection::{open_connection, Connection};
use crate::transport::iterator::RowIterator;
use crate::transport::topology::{Topology, TopologyReader};
use crate::transport::Compression;

type Connections = Vec<(Node, Vec<Arc<Connection>>)>;

const POOL_LOCK_POISONED: &str =
    "Connection pool lock is poisoned, this session is no longer usable. \
                                  Drop it and create a new one.";

pub struct Session {
    // invariant: nonempty
    // PLEASE DO NOT EXPOSE IT TO THE USER OR PEOPLE WILL DIE
    pool: Arc<RwLock<HashMap<Node, NodePool>>>,

    // compression options passed to the session when it was created
    compression: Option<Compression>,

    topology: Topology,
    _topology_reader_handle: RemoteHandle<()>,
}

// Pool of connections for a single node.
// Also stores the sharding info for this node if we have one.
// If we're connecting using a non-shard-aware port, or connecting to Cassandra,
// the pool will always have a single connection under shard 0
// and the sharding info will be empty.
struct NodePool {
    // invariant: nonempty
    connections: HashMap<Shard, Arc<Connection>>,
    shard_info: Option<ShardInfo>,
}

// Trait used to implement Vec<result::Row>::into_typed<RowT>(self)
// This is the only way to add custom method to Vec
pub trait IntoTypedRows {
    fn into_typed<RowT: FromRow>(self) -> TypedRowIter<RowT>;
}

// Adds method Vec<result::Row>::into_typed<RowT>(self)
// It transforms the Vec into iterator mapping to custom row type
impl IntoTypedRows for Vec<result::Row> {
    fn into_typed<RowT: FromRow>(self) -> TypedRowIter<RowT> {
        TypedRowIter {
            row_iter: self.into_iter(),
            phantom_data: Default::default(),
        }
    }
}

// Iterator that maps a Vec<result::Row> into custom RowType used by IntoTypedRows::into_typed
// impl Trait doesn't compile so we have to be explicit
pub struct TypedRowIter<RowT: FromRow> {
    row_iter: std::vec::IntoIter<result::Row>,
    phantom_data: std::marker::PhantomData<RowT>,
}

impl<RowT: FromRow> Iterator for TypedRowIter<RowT> {
    type Item = Result<RowT, FromRowError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.row_iter.next().map(RowT::from_row)
    }
}

/// Represents a CQL session, which can be used to communicate
/// with the database
impl Session {
    // FIXME: Use ToSocketAddrs instead of SocketAddr in public interfaces
    // because it's more convenient
    /// Estabilishes a CQL session with the database
    /// # Arguments
    ///
    /// * `addr` - address of the server
    /// * `compression` - optional compression settings
    pub async fn connect(
        addr: impl ToSocketAddrs + Display,
        compression: Option<Compression>,
    ) -> Result<Self, ConnectionError> {
        let addr = resolve(addr).await?;
        let connection = open_connection(addr, None, compression).await?;
        let node = Node { addr };

        let (topology_reader, topology) = TopologyReader::new(node).await?;
        let (fut, _topology_reader_handle) = topology_reader.run().remote_handle();
        tokio::task::spawn(fut);

        let pool = Arc::new(RwLock::new(
            vec![(node, NodePool::new(Arc::new(connection)))]
                .into_iter()
                .collect(),
        ));

        Ok(Session {
            pool,
            compression,
            topology,
            _topology_reader_handle,
        })
    }

    // TODO: Should return an iterator over results
    // actually, if we consider "INSERT" a query, then no.
    // But maybe "INSERT" and "SELECT" should go through different methods,
    // so we expect "SELECT" to always return Vec<result::Row>?
    /// Sends a query to the database and receives a response.
    /// If `query` has paging enabled, this will return only the first page.
    /// # Arguments
    ///
    /// * `query` - query to be performed
    /// * `values` - values bound to the query
    pub async fn query(
        &self,
        query: impl Into<Query>,
        values: &[Value],
    ) -> Result<Option<Vec<result::Row>>, ConnectionError> {
        self.any_connection()?
            .query_single_page(query, values)
            .await
    }

    pub fn query_iter(
        &self,
        query: impl Into<Query>,
        values: &[Value],
    ) -> Result<RowIterator, ConnectionError> {
        Ok(RowIterator::new_for_query(
            self.any_connection()?,
            query.into(),
            values.to_owned(),
        ))
    }

    /// Prepares a statement on the server side and returns a prepared statement,
    /// which can later be used to perform more efficient queries
    /// # Arguments
    ///#
    pub async fn prepare(&self, query: &str) -> Result<PreparedStatement, ConnectionError> {
        // FIXME: Prepared statement ids are local to a node, so we must make sure
        // that prepare() sends to all nodes and keeps all ids.
        let result = self.any_connection()?.prepare(query.to_owned()).await?;
        match result {
            Response::Error(err) => Err(err.into()),
            Response::Result(result::Result::Prepared(p)) => Ok(PreparedStatement::new(
                p.id,
                p.prepared_metadata,
                query.to_owned(),
            )),
            _ => Err(ConnectionError::UnexpectedFrame),
        }
    }

    /// Executes a previously prepared statement
    /// # Arguments
    ///
    /// * `prepared` - a statement prepared with [prepare](crate::transport::session::prepare)
    /// * `values` - values bound to the query
    pub async fn execute(
        &self,
        prepared: &PreparedStatement,
        values: &[Value],
    ) -> Result<Option<Vec<result::Row>>, ConnectionError> {
        // FIXME: Prepared statement ids are local to a node, so we must make sure
        // that prepare() sends to all nodes and keeps all ids.
        let token = calculate_token(prepared, values);
        let connection = self.pick_connection(token).await?;
        let result = connection.execute(prepared, values, None).await?;
        match result {
            Response::Error(err) => {
                match err.code {
                    9472 => {
                        // Repreparation of a statement is needed
                        let reprepared_result = connection
                            .prepare(prepared.get_statement().to_owned())
                            .await?;
                        // Reprepared statement should keep its id - it's the md5 sum
                        // of statement contents
                        match reprepared_result {
                            Response::Error(err) => return Err(err.into()),
                            Response::Result(result::Result::Prepared(reprepared)) => {
                                if reprepared.id != prepared.get_id() {
                                    return Err(ConnectionError::RepreparedStatmentIDChanged);
                                }
                            }
                            _ => return Err(ConnectionError::UnexpectedFrame),
                        }
                        let result = connection.execute(prepared, values, None).await?;
                        match result {
                            Response::Error(err) => Err(err.into()),
                            Response::Result(result::Result::Rows(rs)) => Ok(Some(rs.rows)),
                            Response::Result(_) => Ok(None),
                            _ => Err(ConnectionError::UnexpectedFrame),
                        }
                    }
                    _ => Err(err.into()),
                }
            }
            Response::Result(result::Result::Rows(rs)) => Ok(Some(rs.rows)),
            Response::Result(_) => Ok(None),
            _ => Err(ConnectionError::UnexpectedFrame),
        }
    }

    pub fn execute_iter(
        &self,
        prepared: impl Into<PreparedStatement>,
        values: &[Value],
    ) -> Result<RowIterator, ConnectionError> {
        Ok(RowIterator::new_for_prepared_statement(
            self.any_connection()?,
            prepared.into(),
            values.to_owned(),
        ))
    }

    /// Sends a batch to the database.
    /// # Arguments
    ///
    /// * `batch` - batch to be performed
    /// * `values` - values bound to the query
    pub async fn batch(
        &self,
        batch: &Batch,
        values: &[impl AsRef<[Value]>],
    ) -> Result<(), ConnectionError> {
        // FIXME: Prepared statement ids are local to a node
        // this method does not handle this
        let response = self.any_connection()?.batch(&batch, values).await?;
        match response {
            Response::Error(err) => Err(err.into()),
            Response::Result(_) => Ok(()),
            _ => Err(ConnectionError::UnexpectedFrame),
        }
    }

    /// Returns all connections that the session has currently opened.
    pub fn get_connections(&self) -> Result<Connections, ConnectionError> {
        Ok(self
            .pool
            .read()
            .map_err(|_| ConnectionError::PoolLockPoisoned(POOL_LOCK_POISONED.to_string()))?
            .iter()
            .map(|(&node, node_pool)| (node, node_pool.get_connections()))
            .collect())
    }

    pub async fn refresh_topology(&self) -> Result<(), ConnectionError> {
        self.topology.refresh().await
    }

    async fn pick_connection(&self, t: Token) -> Result<Arc<Connection>, ConnectionError> {
        // TODO: we try only the owner of the range (vnode) that the token lies in
        // we should calculate the *set* of replicas for this token, using the replication strategy
        // of the table being queried.
        let owner = self.topology.read_ring()?.owner(t);

        let mut found_pool = false;
        let mut shard_info = None;
        if let Some(node_pool) = self
            .pool
            .read()
            .map_err(|_| ConnectionError::PoolLockPoisoned(POOL_LOCK_POISONED.to_string()))?
            .get(&owner)
        {
            // We have a pool for this node. Check if we have a connection for the token.
            if let Some(c) = node_pool.connection_for_token(t) {
                return Ok(c);
            }

            found_pool = true;
            shard_info = node_pool.get_shard_info().cloned();
        };

        // We can't put this block in `else` for the previous if statement, because the block
        // prolongs the lifetime of the read guard from `self.pool.read()`.
        if !found_pool {
            // We don't even have a pool for this node; in particular, we don't know its
            // sharding info. We will just connect to a random shard this time so we don't do
            // too much work on the query path (and we might still get lucky).
            // The next request using this node's pool will pick the right connection, creating
            // it if necessary.
            let new_conn = open_connection(owner.addr, None, self.compression).await?;
            let new_node_pool = NodePool::new(Arc::new(new_conn));

            return Ok(self
                .pool
                .write()
                .map_err(|_| ConnectionError::PoolLockPoisoned(POOL_LOCK_POISONED.to_string()))?
                .entry(owner)
                // If someone created a node pool for this node faster while we were creating
                // our own and inserted it into the session pool, drop our new pool and use the
                // existing one.
                // From this existing pool we will take any connection if there is no
                // connection for this token yet.
                .or_insert(new_node_pool)
                .connection_for_token_or_any(t)?);
        };

        // We had a pool for this node, but there was no connection for this token in that pool.
        // Let's create one.
        // TODO: don't do this on the query path, but concurrently in some other fiber?
        // Take any connection if there is no good connection for this node yet?
        // TODO: opening this connection may actually fail because the port is already in
        // use. We should try binding to a couple of ports before we fail. For that we need
        // open_connection to return a well-typed error so we can check if it was an AddrInUse
        // error. Or refactor this whole thing.
        let new_conn_source_port = shard_info
            .as_ref()
            .map(|info| info.draw_source_port_for_token(t));
        let new_conn =
            Arc::new(open_connection(owner.addr, new_conn_source_port, self.compression).await?);

        Ok(
            match self
                .pool
                .write()
                .map_err(|_| ConnectionError::PoolLockPoisoned(POOL_LOCK_POISONED.to_string()))?
                .entry(owner)
            {
                Entry::Vacant(entry) => {
                    entry.insert(NodePool::new(new_conn.clone()));
                    new_conn
                }
                Entry::Occupied(mut entry) => {
                    let node_pool = entry.get_mut();
                    if shard_info.as_ref() != node_pool.get_shard_info() {
                        // The node has resharded while we were creating the connection.
                        // Take any connection.
                        return node_pool.connection_for_token_or_any(t);
                    }

                    let shard = node_pool.shard_for_token(t);
                    node_pool.or_insert(shard, new_conn)
                }
            },
        )
    }

    fn any_connection(&self) -> Result<Arc<Connection>, ConnectionError> {
        Ok(self
            .pool
            .read()
            // TODO: better error message?
            .map_err(|_| ConnectionError::PoolLockPoisoned(POOL_LOCK_POISONED.to_string()))?
            .values()
            .next()
            .ok_or_else(|| ConnectionError::FatalConnectionError)?
            .any_connection()?)
    }
}

impl NodePool {
    pub fn new(connection: Arc<Connection>) -> Self {
        let (shard, shard_info) =
            match (connection.get_shard_info(), connection.get_is_shard_aware()) {
                (Some(info), true) => (
                    info.shard_of_source_port(connection.get_source_port()),
                    Some(info.to_owned()),
                ),
                _ => (0, None),
            };

        Self {
            connections: vec![(shard, connection)].into_iter().collect(),
            shard_info,
        }
    }

    /// Returns all connections that the session has currently opened.
    pub fn get_connections(&self) -> Vec<Arc<Connection>> {
        self.connections
            .values()
            .map(|conn| conn.to_owned())
            .collect()
    }

    pub fn get_shard_info(&self) -> Option<&ShardInfo> {
        self.shard_info.as_ref()
    }

    pub fn shard_for_token(&self, t: Token) -> Shard {
        self.shard_info.as_ref().map_or(0, |info| info.shard_of(t))
    }

    pub fn connection_for_token(&self, t: Token) -> Option<Arc<Connection>> {
        let shard = self.shard_for_token(t);
        self.connections.get(&shard).map(|c| c.to_owned())
    }

    pub fn connection_for_token_or_any(
        &self,
        t: Token,
    ) -> Result<Arc<Connection>, ConnectionError> {
        self.connection_for_token(t)
            .map_or_else(|| self.any_connection(), Ok)
    }

    pub fn any_connection(&self) -> Result<Arc<Connection>, ConnectionError> {
        Ok(self
            .connections
            .values()
            .next()
            .ok_or_else(|| ConnectionError::FatalConnectionError)?
            .to_owned())
    }

    pub fn or_insert(&mut self, shard: Shard, conn: Arc<Connection>) -> Arc<Connection> {
        self.connections.entry(shard).or_insert(conn).to_owned()
    }
}

fn calculate_token<'a>(stmt: &PreparedStatement, values: &'a [Value]) -> Token {
    // TODO: take the partitioner of the table that is being queried and calculate the token using
    // that partitioner. The below logic gives correct token only for murmur3partitioner
    murmur3_token(stmt.compute_partition_key(values))
}

// Resolve the given `ToSocketAddrs` using a DNS lookup if necessary.
// The resolution may return multiple IPs and the function returns one of them.
// It prefers to return IPv4s first, and only if there are none, IPv6s.
async fn resolve(addr: impl ToSocketAddrs + Display) -> Result<SocketAddr, ConnectionError> {
    let failed_err = ConnectionError::FailedToResolveAddress(addr.to_string());
    let mut ret = None;
    for a in lookup_host(addr).await? {
        match a {
            SocketAddr::V4(_) => return Ok(a),
            _ => {
                ret = Some(a);
            }
        }
    }

    ret.ok_or(failed_err)
}
