pub mod xenomorph {
    pub mod inference {
        include!(concat!(env!("OUT_DIR"), "/xenom.inference.rs"));
    }
    pub mod governance {
        include!(concat!(env!("OUT_DIR"), "/xenom.governance.rs"));
    }
}
