/// Node represents a cluster node along with it's data and connections
use crate::routing::Token;
use crate::transport::connection::Connection;
use crate::transport::connection::VerifiedKeyspaceName;
use crate::transport::connection_pool::{NodeConnectionPool, PoolConfig};
use crate::transport::errors::QueryError;

use std::{
    hash::{Hash, Hasher},
    net::SocketAddr,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

/// Node represents a cluster node along with it's data and connections
pub struct Node {
    pub address: SocketAddr,
    pub datacenter: Option<String>,
    pub rack: Option<String>,

    pool: NodeConnectionPool,

    down_marker: AtomicBool,
}

#[derive(Debug)]
struct UseKeyspaceRequest {
    keyspace_name: VerifiedKeyspaceName,
    response_chan: tokio::sync::oneshot::Sender<Result<(), QueryError>>,
}

impl Node {
    /// Creates new node which starts connecting in the background
    /// # Arguments
    ///
    /// `address` - address to connect to
    /// `compression` - preferred compression to use
    /// `datacenter` - optional datacenter name
    /// `rack` - optional rack name
    pub async fn new(
        address: SocketAddr,
        pool_config: PoolConfig,
        datacenter: Option<String>,
        rack: Option<String>,
        keyspace_name: Option<VerifiedKeyspaceName>,
    ) -> Self {
        let pool =
            NodeConnectionPool::new(address.ip(), address.port(), pool_config, keyspace_name).await;

        Node {
            address,
            datacenter,
            rack,
            pool,
            down_marker: false.into(),
        }
    }

    /// Get connection which should be used to connect using given token
    /// If this connection is broken get any random connection to this Node
    pub async fn connection_for_token(&self, token: Token) -> Result<Arc<Connection>, QueryError> {
        self.pool.connection_for_token(token)
    }

    /// Get random connection
    pub async fn random_connection(&self) -> Result<Arc<Connection>, QueryError> {
        self.pool.random_connection()
    }

    pub fn is_down(&self) -> bool {
        self.down_marker.load(Ordering::Relaxed)
    }

    pub fn change_down_marker(&self, is_down: bool) {
        self.down_marker.store(is_down, Ordering::Relaxed);
    }

    pub async fn use_keyspace(
        &self,
        keyspace_name: VerifiedKeyspaceName,
    ) -> Result<(), QueryError> {
        self.pool.use_keyspace(keyspace_name).await
    }

    pub fn get_working_connections(&self) -> Result<Vec<Arc<Connection>>, QueryError> {
        self.pool.get_working_connections()
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.address == other.address
    }
}

impl Eq for Node {}

impl Hash for Node {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.address.hash(state);
    }
}
