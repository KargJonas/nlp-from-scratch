# Use the official Node image for building
#FROM node:latest AS builder
FROM oven/bun:latest AS frontend-builder

# Create a directory to hold your application code
WORKDIR /frontend

# Copy the application code into the container
COPY . /frontend

RUN ls

# Install dependencies
RUN bun install

# Build your application
RUN bun run build

# Use a smaller, production-ready Node image for running
FROM nginx:latest

# Create a directory for your application
WORKDIR /usr/share/nginx/html

# Copy the built application from the builder stage
RUN rm /usr/share/nginx/html/*
COPY --from=frontend-builder /frontend/dist /usr/share/nginx/html

RUN ls
