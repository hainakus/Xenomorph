

# Use the minimal Ubuntu 22.04 base image
FROM ubuntu:22.04
LABEL authors="hainakus"
# Set working directory
WORKDIR /app/

# Copy the built binary into the container's /app/ directory
COPY . /app/

# Move the picod binary to /usr/local/bin for global access
RUN mv /app/picod /usr/local/bin/picod

# Make sure the picod binary is executable
RUN chmod +x /usr/local/bin/picod

# Expose necessary ports for P2P and RPC communication
EXPOSE 17610 26666

# Set the entrypoint to run the 'picod' binary
ENTRYPOINT ["/usr/local/bin/picod"]

# Default command with necessary flags
CMD ["--utxoindex", "--rpclisten-borsh", "0.0.0.0:17610"]
