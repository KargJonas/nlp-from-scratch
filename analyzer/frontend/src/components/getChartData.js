
function hexToRgb(hex) {
  var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result ? {
    r: parseInt(result[1], 16),
    g: parseInt(result[2], 16),
    b: parseInt(result[3], 16)
  } : null;
}

function getSmoothed(data, n = 20) {

  if (data.length <= n) return [];

  const smoothed = [];

  for (let i = 0; i < data.length - n; i++) {
    const avg = data
      .slice(i, i + n)
      .reduce((acc, val) => acc + Number(val), 0) / n;

    smoothed.push(avg);
  }

  return smoothed;
}

export default function getChartData(rawData, label, rawColor, smoothing = 1) {
  const data = smoothing === 1 ? rawData : getSmoothed(rawData);
  const labels = new Array(data.length)
    .fill(0)
    .map((_, index) => index);

  const { r, g, b } = hexToRgb(rawColor);
  const borderColor = `rgb(${r}, ${g}, ${b})`;
  const backgroundColor = `rgba(${r}, ${g}, ${b}, 0.5)`;

  return {
    labels,
    datasets: [
      {
        label,
        data,
        borderColor,
        backgroundColor,
        pointRadius: 0
      },
    ],
  };
}
