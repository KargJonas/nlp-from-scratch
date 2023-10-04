import { createServer } from "node:http";
import fs from 'node:fs';
import express from 'express';
import { Server } from 'socket.io';

let port = process.env.PORT;
// const METRICS_DIR = '/home/jonas/code/nlp-from-scratch/metrics';
const METRICS_DIR = '/metrics';

if (port === undefined) {
  console.log('$PORT not in ENV. falling back to port 8000.');
  console.log('$PORT should be set by docker-compose. did you start the server some other way?');
  port = 8000;
}

const app = express();
const server = createServer(app);
const io = new Server(server, {
  cors: { origin: ['http://localhost:5000', 'localhost:5000'] }
});

const files = fs.readdirSync(METRICS_DIR);
console.log(files)

io.on('connection', (socket) => {
  console.log('a user connected');

  socket.on('open-file', (file) => {
    console.log('sending data');

    const csv = fs.readFileSync(`${METRICS_DIR}/${file}`, 'utf8');
    socket.emit('data', csv);
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

server.listen(port, () => {
  console.log(`server running at http://localhost:${port}`);
});
