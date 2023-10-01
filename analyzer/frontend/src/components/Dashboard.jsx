import './Dashboard.scss';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import getChartData from "./getChartData.js";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const options = {
  // maintainAspectRatio: true,
  responsive: true,
  plugins: {
    legend: { position: 'top' },
    title: {
      display: true,
      // text: '',
    },
  },
};

export default function Dashboard({ data, file }) {
  if (!data) return;

  const chartData = getChartData(data, 'Batch loss', '#3D4366');
  const chartDataSmoothed = getChartData(data, 'Batch loss [smoothed]', '#8CC084', 20);

  return (
    <div className='Dashboard'>
      <h1>Dashboard</h1>
      <div className='chart-container'>
        <Line
          options={options}
          data={chartDataSmoothed}
          width='100%'
          height='70%'
        />

        <Line
          options={options}
          data={chartData}
          width='100%'
          height='70%'
        />
      </div>
    </div>
  )
}
