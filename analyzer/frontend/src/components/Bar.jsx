import './Bar.scss';
import classNames from "classnames";
import moment from "moment";

function List({ files, activeFile, fileClicked }) {

  const lis = files.map((file, index) => {
    const [name, stamp] = file.split('.');
    const active = file === activeFile ? 'active' : null;
    const creationTime = moment(Number(stamp)).fromNow().replace('minutes', 'min');

    return (
      <li
        className={classNames('list', active)}
        key={index}
        onClick={() => fileClicked(file)}
      >
        <span className='file-name'>{name}</span>
        <span className='creation-date'>{creationTime}</span>
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
