use crate::routing::hash3_x64_128;
use crate::transport::session::Session;

// TODO: Requires a running local Scylla instance
#[tokio::test]
#[ignore]
async fn test_unprepared_statement() {
    let session = Session::connect("127.0.0.1:9042", None).await.unwrap();

    session.query("CREATE KEYSPACE IF NOT EXISTS ks WITH REPLICATION = {'class' : 'SimpleStrategy', 'replication_factor' : 1}", &[]).await.unwrap();
    session
        .query("DROP TABLE IF EXISTS ks.t;", &[])
        .await
        .unwrap();
    session
        .query(
            "CREATE TABLE IF NOT EXISTS ks.t (a int, b int, c text, primary key (a, b))",
            &[],
        )
        .await
        .unwrap();
    // Wait for schema agreement
    std::thread::sleep(std::time::Duration::from_millis(300));
    session
        .query("INSERT INTO ks.t (a, b, c) VALUES (1, 2, 'abc')", &[])
        .await
        .unwrap();
    session
        .query("INSERT INTO ks.t (a, b, c) VALUES (7, 11, '')", &[])
        .await
        .unwrap();
    session
        .query("INSERT INTO ks.t (a, b, c) VALUES (1, 4, 'hello')", &[])
        .await
        .unwrap();

    let rs = session
        .query("SELECT a, b, c FROM ks.t", &[])
        .await
        .unwrap()
        .unwrap();

    let mut results: Vec<(i32, i32, &String)> = rs
        .iter()
        .map(|r| {
            let a = r.columns[0].as_ref().unwrap().as_int().unwrap();
            let b = r.columns[1].as_ref().unwrap().as_int().unwrap();
            let c = r.columns[2].as_ref().unwrap().as_text().unwrap();
            (a, b, c)
        })
        .collect();
    results.sort();
    assert_eq!(
        results,
        vec![
            (1, 2, &String::from("abc")),
            (1, 4, &String::from("hello")),
            (7, 11, &String::from(""))
        ]
    );
}

#[tokio::test]
#[ignore]
async fn test_prepared_statement() {
    let session = Session::connect("127.0.0.1:9042", None).await.unwrap();

    session.query("CREATE KEYSPACE IF NOT EXISTS ks WITH REPLICATION = {'class' : 'SimpleStrategy', 'replication_factor' : 1}", &[]).await.unwrap();
    session
        .query("DROP TABLE IF EXISTS ks.t2;", &[])
        .await
        .unwrap();
    session
        .query("DROP TABLE IF EXISTS ks.complex_pk;", &[])
        .await
        .unwrap();
    session
        .query(
            "CREATE TABLE IF NOT EXISTS ks.t2 (a int, b int, c text, primary key (a, b))",
            &[],
        )
        .await
        .unwrap();
    session
        .query("CREATE TABLE IF NOT EXISTS ks.complex_pk (a int, b int, c text, d int, e int, primary key ((a,b,c),d))", &[])
        .await
        .unwrap();
    // Wait for schema agreement
    std::thread::sleep(std::time::Duration::from_millis(300));
    let prepared_statement = session
        .prepare("INSERT INTO ks.t2 (a, b, c) VALUES (?, ?, ?)")
        .await
        .unwrap();
    let prepared_complex_pk_statement = session
        .prepare("INSERT INTO ks.complex_pk (a, b, c, d) VALUES (?, ?, ?, 7)")
        .await
        .unwrap();

    let values = values!(17_i32, 16_i32, "I'm prepared!!!");

    session.execute(&prepared_statement, &values).await.unwrap();
    session
        .execute(&prepared_complex_pk_statement, &values)
        .await
        .unwrap();

    // Verify that token calculation is compatible with Scylla
    {
        let rs = session
            .query("SELECT token(a) FROM ks.t2", &[])
            .await
            .unwrap()
            .unwrap();
        let token: i64 = rs.first().unwrap().columns[0]
            .as_ref()
            .unwrap()
            .as_bigint()
            .unwrap();
        let expected_token =
            hash3_x64_128(&prepared_statement.compute_partition_key(&values)) as i64;
        assert_eq!(token, expected_token)
    }
    {
        let rs = session
            .query("SELECT token(a,b,c) FROM ks.complex_pk", &[])
            .await
            .unwrap()
            .unwrap();
        let token: i64 = rs.first().unwrap().columns[0]
            .as_ref()
            .unwrap()
            .as_bigint()
            .unwrap();
        let expected_token =
            hash3_x64_128(&prepared_complex_pk_statement.compute_partition_key(&values)) as i64;
        assert_eq!(token, expected_token)
    }

    // Verify that correct data was insertd
    {
        let rs = session
            .query("SELECT a,b,c FROM ks.t2", &[])
            .await
            .unwrap()
            .unwrap();
        let r = rs.first().unwrap();
        let a = r.columns[0].as_ref().unwrap().as_int().unwrap();
        let b = r.columns[1].as_ref().unwrap().as_int().unwrap();
        let c = r.columns[2].as_ref().unwrap().as_text().unwrap();
        assert_eq!((a, b, c), (17, 16, &String::from("I'm prepared!!!")))
    }
    {
        let rs = session
            .query("SELECT a,b,c,d,e FROM ks.complex_pk", &[])
            .await
            .unwrap()
            .unwrap();
        let r = rs.first().unwrap();
        let a = r.columns[0].as_ref().unwrap().as_int().unwrap();
        let b = r.columns[1].as_ref().unwrap().as_int().unwrap();
        let c = r.columns[2].as_ref().unwrap().as_text().unwrap();
        let d = r.columns[3].as_ref().unwrap().as_int().unwrap();
        let e = r.columns[4].as_ref();
        assert!(e.is_none());
        assert_eq!((a, b, c, d), (17, 16, &String::from("I'm prepared!!!"), 7))
    }
}

#[tokio::test]
#[ignore]
async fn test_batch() {
    let session = Session::connect("127.0.0.1:9042", None).await.unwrap();

    session.query("CREATE KEYSPACE IF NOT EXISTS ks WITH REPLICATION = {'class' : 'SimpleStrategy', 'replication_factor' : 1}", &[]).await.unwrap();
    session
        .query("DROP TABLE IF EXISTS ks.t;", &[])
        .await
        .unwrap();
    session
        .query(
            "CREATE TABLE IF NOT EXISTS ks.t (a int, b int, c text, primary key (a, b))",
            &[],
        )
        .await
        .unwrap();

    // Wait for schema agreement
    std::thread::sleep(std::time::Duration::from_millis(300));

    let prepared_statement = session
        .prepare("INSERT INTO ks.t (a, b, c) VALUES (?, ?, ?)")
        .await
        .unwrap();

    // TODO: Add API, that supports binding values to statements in batch creation process,
    // to avoid problem of statements/values count mismatch
    use crate::batch::Batch;
    let mut batch: Batch = Default::default();
    batch.append_statement("INSERT INTO ks.t (a, b, c) VALUES (?, ?, ?)");
    batch.append_statement("INSERT INTO ks.t (a, b, c) VALUES (7, 11, '')");
    batch.append_statement(prepared_statement);

    let values: &[&[_]] = &[
        &values!(1_i32, 2_i32, "abc"),
        &[],
        &values!(1_i32, 4_i32, "hello"),
    ];

    session.batch(&batch, values).await.unwrap();

    let rs = session
        .query("SELECT a, b, c FROM ks.t", &[])
        .await
        .unwrap()
        .unwrap();

    let mut results: Vec<(i32, i32, &String)> = rs
        .iter()
        .map(|r| {
            let a = r.columns[0].as_ref().unwrap().as_int().unwrap();
            let b = r.columns[1].as_ref().unwrap().as_int().unwrap();
            let c = r.columns[2].as_ref().unwrap().as_text().unwrap();
            (a, b, c)
        })
        .collect();
    results.sort();
    assert_eq!(
        results,
        vec![
            (1, 2, &String::from("abc")),
            (1, 4, &String::from("hello")),
            (7, 11, &String::from(""))
        ]
    );
}

#[tokio::test]
#[ignore]
async fn test_token_calculation() {
    let session = Session::connect("127.0.0.1:9042", None).await.unwrap();

    session.query("CREATE KEYSPACE IF NOT EXISTS ks WITH REPLICATION = {'class' : 'SimpleStrategy', 'replication_factor' : 1}", &[]).await.unwrap();
    session
        .query("DROP TABLE IF EXISTS ks.t3;", &[])
        .await
        .unwrap();
    session
        .query("CREATE TABLE IF NOT EXISTS ks.t3 (a text primary key)", &[])
        .await
        .unwrap();
    // Wait for schema agreement
    std::thread::sleep(std::time::Duration::from_millis(300));
    let prepared_statement = session
        .prepare("INSERT INTO ks.t3 (a) VALUES (?)")
        .await
        .unwrap();

    // Try calculating tokens for different sizes of the key
    for i in 1..50usize {
        eprintln!("Trying key size {}", i);
        let mut s = String::new();
        for _ in 0..i {
            s.push('a');
        }
        let values = values!(s.as_ref());
        session.execute(&prepared_statement, &values).await.unwrap();

        let rs = session
            .query("SELECT token(a) FROM ks.t3 WHERE a = ?", &values)
            .await
            .unwrap()
            .unwrap();
        let token: i64 = rs.first().unwrap().columns[0]
            .as_ref()
            .unwrap()
            .as_bigint()
            .unwrap();
        let expected_token =
            hash3_x64_128(&prepared_statement.compute_partition_key(&values)) as i64;
        assert_eq!(token, expected_token)
    }
}
