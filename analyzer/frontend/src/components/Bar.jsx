import './Bar.scss';
import classNames from "classnames";

function List({ files, activeFile, fileClicked }) {

  const lis = files.map((file, index) => {
    const active = file === activeFile
      ? 'active' : null;

    return (
      <li
        className={classNames('list', active)}
        key={index}
        onClick={() => fileClicked(file)}
      >
        {file}
      </li>
    );
  });

  return <ul className='list'>{lis}</ul>;
}

export default function Bar({ files, activeFile, fileClicked }) {

  return (
    <div className='Bar'>
      <div className='fixed-container'>
        <h2>Collected metrics</h2>
        <List files={files} activeFile={activeFile} fileClicked={fileClicked} />
      </div>
    </div>
  );
}
