import { createServer } from "node:http";
import fs from 'node:fs';
import express from 'express';
import { Server } from 'socket.io';

const METRICS_DIR = '/home/jonas/code/nlp-from-scratch/metrics';

const app = express();
const server = createServer(app);
const io = new Server(server, {
  cors: { origin: 'http://localhost:5173' }
});

io.on('connection', (socket) => {
  console.log('a user connected');

  // Read files in metrics dir and send to frontend
  const files = fs.readdirSync(METRICS_DIR);
  socket.emit('files', files);

  socket.on('open-file', (file) => {
    console.log('sending data');

    const csv = fs.readFileSync(`${METRICS_DIR}/${file}`, 'utf8');
    socket.emit('data', csv);
  });
});

server.listen(3000, () => {
  console.log('server running at http://localhost:3000');
});
