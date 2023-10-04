# Building the backend
FROM oven/bun:latest AS backend-builder

# Create a directory to hold your application code
WORKDIR /backend

# Copy the application code into the container
COPY ./backend /backend

# Install dependencies
RUN bun install

# Building the frontend
FROM oven/bun:latest AS frontend-builder

# Create a directory to hold your application code
WORKDIR /frontend

# Copy the application code into the container
COPY ./frontend /frontend

RUN ls

# Install dependencies
RUN bun install

# Build your application
RUN bun run build

# Deploy frontend and backend
FROM oven/bun:latest

# Create a directory for your application
WORKDIR /app

# Copy the built application from the builder stage
COPY --from=backend-builder /backend /app
COPY --from=frontend-builder /frontend/dist /app/dist

EXPOSE 5000/tcp
EXPOSE 4000/tcp

# Specify the command to run your Node.js application
CMD ["bun", "run", "start"]
