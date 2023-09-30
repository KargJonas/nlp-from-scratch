import './App.scss'
import Bar from "./Bar.jsx";
import {useEffect, useState} from "react";
import {io, Socket} from "socket.io-client";
import Dashboard from "./Dashboard.jsx";

let socket: Socket;

function cleanFiles(files) {
  return files
    .filter(file => file.endsWith('.csv'))
    .sort()
    .reverse();
}

function cleanData(data) {
  return data.trim().split('\n');
}

export default function App() {

  const [files, setFiles] = useState([]);
  const [activeFile, setActiveFile] = useState();
  const [data, setData] = useState([]);

  useEffect(() => {
    socket = io('localhost:3000');

    socket.on('files', data => {
      const cleanedFiles = cleanFiles(data);
      const active = cleanedFiles[0];

      setFiles(cleanedFiles);
      handleClick(active);
    });

    socket.on('data', (data) => setData(cleanData(data)));
  }, []);

  const handleClick = (file) => {
    setActiveFile(file);
    socket?.emit('open-file', file);
  };

  return (
    <div className='App'>
      <div className='h-container'>
        <Bar files={files} activeFile={activeFile} fileClicked={handleClick} />
        <Dashboard data={data} file={activeFile} />
      </div>
    </div>
  )
}
