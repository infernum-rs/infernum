//! HTTP server implementation
//!
//! Provides the [`Server`] builder for registering models and running
//! the OpenAI-compatible API.

use std::net::SocketAddr;

/// Entry point for registering a model with the server.
///
/// Type-erases the model, tokenizer, and chat template so the server
/// doesn't need to know concrete types.
pub struct ModelEntry {
    _name: String,
}

impl ModelEntry {
    /// Create a new model entry (placeholder — full impl in Step 4).
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            _name: name.to_string(),
        }
    }
}

/// The HTTP server.
pub struct Server {
    _bind_addr: SocketAddr,
}

/// Builder for constructing a [`Server`].
pub struct ServerBuilder {
    bind_addr: SocketAddr,
}

impl Server {
    /// Create a new server builder.
    #[must_use]
    pub fn builder() -> ServerBuilder {
        ServerBuilder {
            bind_addr: SocketAddr::from(([0, 0, 0, 0], 8080)),
        }
    }
}

impl ServerBuilder {
    /// Set the address to bind the server to.
    #[must_use]
    pub fn bind(mut self, addr: impl Into<SocketAddr>) -> Self {
        self.bind_addr = addr.into();
        self
    }

    /// Build the server.
    #[must_use]
    pub fn build(self) -> Server {
        Server {
            _bind_addr: self.bind_addr,
        }
    }
}
