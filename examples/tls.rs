use anyhow::Result;
use scylla::cql_to_rust::FromRow;
use scylla::macros::FromRow;
use scylla::transport::session::{IntoTypedRows, Session};
use scylla::SessionBuilder;
use std::env;
use std::fs;
use std::path::PathBuf;

use openssl::ssl::{SslContextBuilder, SslFiletype, SslMethod, SslVerifyMode};

// How to run scylla instance with TLS:
//
// Edit your scylla.yaml file and add paths to certificates
// ex:
// client_encryption_options:
//     enabled: true
//     certificate: /etc/scylla/db.crt
//     keyfile: /etc/scylla/db.key
//
// If using docker mount your scylla.yaml file and your cert files with option
// --volume $(pwd)/tls.yaml:/etc/scylla/scylla.yaml
//
// If python returns permission error 13 use "Z" flag
// --volume $(pwd)/tls.yaml:/etc/scylla/scylla.yaml:Z
//
// In your Rust program connect to port 9142 if it wasn't changed
// Create new SslContextBuilder with SslMethod that is used in your connection
// Use set_certificate_file method with path to your .crt file and its filetype as arguments
// Set verification mode
// Build it and add to scylla-rust-driver's SessionBuilder
#[tokio::main]
async fn main() -> Result<()> {
    // Create connection
    let uri = env::var("SCYLLA_URI").unwrap_or_else(|_| "127.0.0.1:9142".to_string());

    println!("Connecting to {} ...", uri);

    let certdir = fs::canonicalize(PathBuf::from("./examples/certs/scylla.crt"))?;
    let mut context_builder = SslContextBuilder::new(SslMethod::tls())?;
    context_builder.set_certificate_file(certdir.as_path(), SslFiletype::PEM)?;
    context_builder.set_verify(SslVerifyMode::NONE);

    let session: Session = SessionBuilder::new()
        .known_node(uri)
        .ssl_context(Some(context_builder.build()))
        .build()
        .await?;

    session.query("CREATE KEYSPACE IF NOT EXISTS ks WITH REPLICATION = {'class' : 'SimpleStrategy', 'replication_factor' : 1}", &[]).await?;

    session
        .query(
            "CREATE TABLE IF NOT EXISTS ks.t (a int, b int, c text, primary key (a, b))",
            &[],
        )
        .await?;

    session
        .query("INSERT INTO ks.t (a, b, c) VALUES (?, ?, ?)", (3, 4, "def"))
        .await?;

    session
        .query("INSERT INTO ks.t (a, b, c) VALUES (1, 2, 'abc')", &[])
        .await?;

    let prepared = session
        .prepare("INSERT INTO ks.t (a, b, c) VALUES (?, 7, ?)")
        .await?;
    session
        .execute(&prepared, (42_i32, "I'm prepared!"))
        .await?;
    session
        .execute(&prepared, (43_i32, "I'm prepared 2!"))
        .await?;
    session
        .execute(&prepared, (44_i32, "I'm prepared 3!"))
        .await?;

    // Rows can be parsed as tuples
    if let Some(rows) = session.query("SELECT a, b, c FROM ks.t", &[]).await? {
        for row in rows.into_typed::<(i32, i32, String)>() {
            let (a, b, c) = row?;
            println!("a, b, c: {}, {}, {}", a, b, c);
        }
    }

    // Or as custom structs that derive FromRow
    #[derive(Debug, FromRow)]
    struct RowData {
        a: i32,
        b: Option<i32>,
        c: String,
    }

    if let Some(rows) = session.query("SELECT a, b, c FROM ks.t", &[]).await? {
        for row_data in rows.into_typed::<RowData>() {
            let row_data = row_data?;
            println!("row_data: {:?}", row_data);
        }
    }

    // Or simply as untyped rows
    if let Some(rows) = session.query("SELECT a, b, c FROM ks.t", &[]).await? {
        for row in rows {
            let a = row.columns[0].as_ref().unwrap().as_int().unwrap();
            let b = row.columns[1].as_ref().unwrap().as_int().unwrap();
            let c = row.columns[2].as_ref().unwrap().as_text().unwrap();
            println!("a, b, c: {}, {}, {}", a, b, c);

            // Alternatively each row can be parsed individually
            // let (a2, b2, c2) = row.into_typed::<(i32, i32, String)>() ?;
        }
    }

    let metrics = session.get_metrics();
    println!("Queries requested: {}", metrics.get_queries_num());
    println!("Iter queries requested: {}", metrics.get_queries_iter_num());
    println!("Errors occured: {}", metrics.get_errors_num());
    println!("Iter errors occured: {}", metrics.get_errors_iter_num());
    println!("Average latency: {}", metrics.get_latency_avg_ms().unwrap());
    println!(
        "99.9 latency percentile: {}",
        metrics.get_latency_percentile_ms(99.9).unwrap()
    );

    println!("Ok.");

    Ok(())
}