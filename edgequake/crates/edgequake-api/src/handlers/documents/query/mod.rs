//! Document query handlers — split by SRP.

pub mod detail;
pub mod figure_media;
pub mod list;
pub mod scan;
pub mod table;
pub mod track_status;

pub use detail::*;
pub use figure_media::*;
pub use list::*;
pub use scan::*;
pub use table::*;
pub use track_status::*;
