

# Use the minimal Ubuntu 22.04 base image
FROM ubuntu:22.04
LABEL authors="hainakus"
# Set working directory
WORKDIR /app/

# Copy the built binary into the container's /app/ directory
COPY . /app/

# Move the xenom binary to /usr/local/bin for global access
RUN mv /app/xenom /usr/local/bin/xenom

# Make sure the xenom binary is executable
RUN chmod +x /usr/local/bin/xenom

# Expose necessary ports for P2P and RPC communication
EXPOSE 17610 26666

# Set the entrypoint to run the 'xenom' binary
ENTRYPOINT ["/usr/local/bin/xenon"]

# Default command with necessary flags
CMD ["--utxoindex", "--rpclisten-borsh", "0.0.0.0:17610"]
