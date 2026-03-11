use anyhow::Result;
use sqlx::{Pool, Row, Sqlite, SqlitePool};
use std::sync::Arc;

// ── Row types returned to callers ─────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DbMiner {
    pub worker:        String,
    pub address:       String,
    pub first_seen:    i64,
    pub last_share:    i64,
    pub shares_total:  i64,
    pub blocks_total:  i64,
    pub current_diff:  f64,
    pub hashrate_hps:  f64,
    pub connected:     bool,
}

#[derive(Debug, Clone)]
pub struct DbBlock {
    pub job_id:          String,
    pub found_at:        i64,
    pub block_daa_score: i64,
    pub status:          String,
    pub tx_id:           Option<String>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct DbBlockPayout {
    pub job_id:     String,
    pub worker:     String,
    pub proportion: f64,
}

// ── Db handle ─────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct Db(Arc<Pool<Sqlite>>);

impl Db {
    /// Open (or create) the SQLite database at `path` and run schema migrations.
    pub async fn open(path: &str) -> Result<Self> {
        let url  = format!("sqlite:{path}?mode=rwc");
        let pool = SqlitePool::connect(&url).await?;
        let db   = Self(Arc::new(pool));
        db.migrate().await?;
        Ok(db)
    }

    async fn migrate(&self) -> Result<()> {
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS miners (
                worker       TEXT    PRIMARY KEY,
                address      TEXT    NOT NULL,
                first_seen   INTEGER NOT NULL,
                last_share   INTEGER NOT NULL DEFAULT 0,
                shares_total INTEGER NOT NULL DEFAULT 0,
                blocks_total INTEGER NOT NULL DEFAULT 0,
                current_diff REAL    NOT NULL DEFAULT 1.0,
                hashrate_hps REAL    NOT NULL DEFAULT 0.0,
                connected    INTEGER NOT NULL DEFAULT 0
            )",
        )
        .execute(self.pool())
        .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS shares (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                worker       TEXT    NOT NULL,
                job_id       TEXT    NOT NULL,
                difficulty   REAL    NOT NULL,
                submitted_at INTEGER NOT NULL
            )",
        )
        .execute(self.pool())
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_shares_worker ON shares(worker)",
        )
        .execute(self.pool())
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_shares_at ON shares(submitted_at)",
        )
        .execute(self.pool())
        .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS blocks (
                job_id          TEXT    PRIMARY KEY,
                found_at        INTEGER NOT NULL,
                block_daa_score INTEGER NOT NULL,
                status          TEXT    NOT NULL DEFAULT 'pending',
                tx_id           TEXT
            )",
        )
        .execute(self.pool())
        .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS block_payouts (
                job_id     TEXT NOT NULL,
                worker     TEXT NOT NULL,
                proportion REAL NOT NULL,
                PRIMARY KEY (job_id, worker)
            )",
        )
        .execute(self.pool())
        .await?;

        Ok(())
    }

    fn pool(&self) -> &Pool<Sqlite> {
        &self.0
    }

    // ── Miner operations ──────────────────────────────────────────────────────

    /// Register a newly authorized miner (upsert preserving totals).
    pub async fn upsert_miner_connected(
        &self,
        worker:  &str,
        address: &str,
        now:     i64,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO miners (worker, address, first_seen, last_share, shares_total, blocks_total, current_diff, hashrate_hps, connected)
             VALUES (?1, ?2, ?3, ?4, 0, 0, 1.0, 0.0, 1)
             ON CONFLICT(worker) DO UPDATE SET
                connected  = 1,
                address    = excluded.address",
        )
        .bind(worker)
        .bind(address)
        .bind(now)
        .bind(now)
        .execute(self.pool())
        .await?;
        Ok(())
    }

    /// Update miner stats on every valid share submission.
    pub async fn upsert_miner_share(
        &self,
        worker:       &str,
        now:          i64,
        current_diff: f64,
        hashrate_hps: f64,
    ) -> Result<()> {
        sqlx::query(
            "UPDATE miners SET
                last_share   = ?2,
                shares_total = shares_total + 1,
                current_diff = ?3,
                hashrate_hps = ?4
             WHERE worker = ?1",
        )
        .bind(worker)
        .bind(now)
        .bind(current_diff)
        .bind(hashrate_hps)
        .execute(self.pool())
        .await?;
        Ok(())
    }

    /// Increment blocks_found counter for a worker.
    pub async fn upsert_miner_block(&self, worker: &str) -> Result<()> {
        sqlx::query(
            "UPDATE miners SET blocks_total = blocks_total + 1 WHERE worker = ?1",
        )
        .bind(worker)
        .execute(self.pool())
        .await?;
        Ok(())
    }

    /// Mark a miner as disconnected; preserve all counters.
    pub async fn set_miner_disconnected(&self, worker: &str) -> Result<()> {
        sqlx::query("UPDATE miners SET connected = 0 WHERE worker = ?1")
            .bind(worker)
            .execute(self.pool())
            .await?;
        Ok(())
    }

    // ── Share operations ──────────────────────────────────────────────────────

    pub async fn insert_share(
        &self,
        worker:  &str,
        job_id:  &str,
        diff:    f64,
        now:     i64,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO shares (worker, job_id, difficulty, submitted_at) VALUES (?1,?2,?3,?4)",
        )
        .bind(worker)
        .bind(job_id)
        .bind(diff)
        .bind(now)
        .execute(self.pool())
        .await?;
        Ok(())
    }

    /// Prune shares older than `max_age_secs` to keep the table bounded.
    #[allow(dead_code)]
    pub async fn prune_shares(&self, max_age_secs: i64, now: i64) -> Result<()> {
        sqlx::query("DELETE FROM shares WHERE submitted_at < ?1")
            .bind(now - max_age_secs)
            .execute(self.pool())
            .await?;
        Ok(())
    }

    // ── Block operations ──────────────────────────────────────────────────────

    pub async fn insert_block(
        &self,
        job_id:    &str,
        found_at:  i64,
        daa_score: i64,
    ) -> Result<()> {
        sqlx::query(
            "INSERT OR IGNORE INTO blocks (job_id, found_at, block_daa_score, status) VALUES (?1,?2,?3,'pending')",
        )
        .bind(job_id)
        .bind(found_at)
        .bind(daa_score)
        .execute(self.pool())
        .await?;
        Ok(())
    }

    pub async fn insert_block_payouts(
        &self,
        job_id:      &str,
        proportions: &[(String, f64)],
    ) -> Result<()> {
        for (worker, proportion) in proportions {
            sqlx::query(
                "INSERT OR IGNORE INTO block_payouts (job_id, worker, proportion) VALUES (?1,?2,?3)",
            )
            .bind(job_id)
            .bind(worker)
            .bind(proportion)
            .execute(self.pool())
            .await?;
        }
        Ok(())
    }

    pub async fn update_block_status(
        &self,
        job_id: &str,
        status: &str,
        tx_id:  Option<&str>,
    ) -> Result<()> {
        sqlx::query(
            "UPDATE blocks SET status = ?2, tx_id = ?3 WHERE job_id = ?1",
        )
        .bind(job_id)
        .bind(status)
        .bind(tx_id)
        .execute(self.pool())
        .await?;
        Ok(())
    }

    // ── Read queries for the API ──────────────────────────────────────────────

    pub async fn get_all_miners(&self) -> Result<Vec<DbMiner>> {
        let rows = sqlx::query(
            "SELECT worker, address, first_seen, last_share, shares_total, blocks_total,
                    current_diff, hashrate_hps, connected
             FROM miners ORDER BY shares_total DESC",
        )
        .fetch_all(self.pool())
        .await?;

        Ok(rows
            .iter()
            .map(|r| DbMiner {
                worker:       r.get("worker"),
                address:      r.get("address"),
                first_seen:   r.get("first_seen"),
                last_share:   r.get("last_share"),
                shares_total: r.get("shares_total"),
                blocks_total: r.get("blocks_total"),
                current_diff: r.get("current_diff"),
                hashrate_hps: r.get("hashrate_hps"),
                connected:    r.get::<i64, _>("connected") != 0,
            })
            .collect())
    }

    pub async fn get_miner(&self, worker_or_addr: &str) -> Result<Option<DbMiner>> {
        let row = sqlx::query(
            "SELECT worker, address, first_seen, last_share, shares_total, blocks_total,
                    current_diff, hashrate_hps, connected
             FROM miners WHERE worker = ?1 OR address = ?1 LIMIT 1",
        )
        .bind(worker_or_addr)
        .fetch_optional(self.pool())
        .await?;

        Ok(row.map(|r| DbMiner {
            worker:       r.get("worker"),
            address:      r.get("address"),
            first_seen:   r.get("first_seen"),
            last_share:   r.get("last_share"),
            shares_total: r.get("shares_total"),
            blocks_total: r.get("blocks_total"),
            current_diff: r.get("current_diff"),
            hashrate_hps: r.get("hashrate_hps"),
            connected:    r.get::<i64, _>("connected") != 0,
        }))
    }

    pub async fn get_blocks(&self, limit: i64) -> Result<Vec<DbBlock>> {
        let rows = sqlx::query(
            "SELECT job_id, found_at, block_daa_score, status, tx_id
             FROM blocks ORDER BY found_at DESC LIMIT ?1",
        )
        .bind(limit)
        .fetch_all(self.pool())
        .await?;

        Ok(rows
            .iter()
            .map(|r| DbBlock {
                job_id:          r.get("job_id"),
                found_at:        r.get("found_at"),
                block_daa_score: r.get("block_daa_score"),
                status:          r.get("status"),
                tx_id:           r.get("tx_id"),
            })
            .collect())
    }

    pub async fn get_block_payouts(&self, job_id: &str) -> Result<Vec<DbBlockPayout>> {
        let rows = sqlx::query(
            "SELECT job_id, worker, proportion FROM block_payouts WHERE job_id = ?1",
        )
        .bind(job_id)
        .fetch_all(self.pool())
        .await?;

        Ok(rows
            .iter()
            .map(|r| DbBlockPayout {
                job_id:     r.get("job_id"),
                worker:     r.get("worker"),
                proportion: r.get("proportion"),
            })
            .collect())
    }

    pub async fn count_blocks(&self) -> Result<i64> {
        let row = sqlx::query("SELECT COUNT(*) AS n FROM blocks")
            .fetch_one(self.pool())
            .await?;
        Ok(row.get("n"))
    }

    pub async fn count_shares(&self) -> Result<i64> {
        let row = sqlx::query("SELECT COUNT(*) AS n FROM shares")
            .fetch_one(self.pool())
            .await?;
        Ok(row.get("n"))
    }

    pub async fn total_pool_hashrate(&self) -> Result<f64> {
        let row = sqlx::query(
            "SELECT COALESCE(SUM(hashrate_hps), 0.0) AS h FROM miners WHERE connected = 1",
        )
        .fetch_one(self.pool())
        .await?;
        Ok(row.get("h"))
    }

    pub async fn count_connected_miners(&self) -> Result<i64> {
        let row = sqlx::query("SELECT COUNT(*) AS n FROM miners WHERE connected = 1")
            .fetch_one(self.pool())
            .await?;
        Ok(row.get("n"))
    }
}
