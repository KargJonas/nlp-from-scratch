import { createServer } from "node:http";
import fs from 'node:fs';
import express from 'express';
import { Server } from 'socket.io';

const frontendPort = process.env.FRONTEND_PORT || 5000;
const backendPort = process.env.BACKEND_PORT || 4000;
const METRICS_DIR = '/metrics';
const DIST_DIR = './dist';

const app = express();
const server = createServer(app);

const io = new Server(server, {
  cors: { origin: [`http://localhost:${frontendPort}`] }
});

console.log('Metrics dir: ' + fs.readdirSync(METRICS_DIR));
console.log('Dist dir: ' + fs.readdirSync(DIST_DIR));

app.use(express.static(DIST_DIR));
app.listen(frontendPort, () => console.log(`frontend running at http://localhost:${frontendPort}`));

io.on('connection', (socket) => {
  console.log('a user connected');

  socket.on('open-file', (file) => {
    console.log('sending data');

    // const csv = fs.readFileSync(`${METRICS_DIR}/${file}`, 'utf8');
    // socket.emit('data', csv);

    try {
      const csv = fs.readFileSync(`${METRICS_DIR}/${file}`, 'utf8');
      socket.emit('data', csv);
    } catch (error) {
      console.error(`error while reading file "${file}": ${error.message}`);
    }
  });

  const handleFilesChanged = () => {
    console.log('files in directory changed, sending data')

    // Read files in metrics dir and send to frontend
    const files = fs.readdirSync(METRICS_DIR);
    socket.emit('files', files);
  }

  fs.watch(METRICS_DIR, handleFilesChanged);
  handleFilesChanged();
});

server.listen(backendPort, () => {
  console.log(`backend running at http://localhost:${backendPort}`);
});
