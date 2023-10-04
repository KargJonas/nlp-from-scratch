# Use the official Node image for building
#FROM node:latest AS builder
FROM oven/bun:latest AS backend-builder

# Create a directory to hold your application code
WORKDIR /app

# Copy the application code into the container
COPY . /app

# Install dependencies
RUN bun install

# Use a smaller, production-ready Node image for running
FROM oven/bun:latest

# Create a directory for your application
WORKDIR /app

# Copy the built application from the builder stage
COPY --from=backend-builder /app /app

# Specify the command to run your Node.js application
CMD ["bun", "run", "start"]
