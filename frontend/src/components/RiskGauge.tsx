import React from 'react';

interface RiskGaugeProps {
  probability: number; // 0 to 1
  size?: number;
}

export const RiskGauge: React.FC<RiskGaugeProps> = ({ probability, size = 200 }) => {
  const percentage = Math.round(probability * 100);
  
  let color = '#22c55e'; // green (<30)
  let category = 'Low Risk';
  if (percentage >= 30 && percentage < 60) {
    color = '#eab308'; // yellow
    category = 'Moderate Risk';
  } else if (percentage >= 60 && percentage < 80) {
    color = '#f97316'; // orange
    category = 'Higher Risk';
  } else if (percentage >= 80) {
    color = '#ef4444'; // red
    category = 'High Risk';
  }

  const radius = (size - 20) / 2;
  const circumference = radius * 2 * Math.PI;
  const strokeDashoffset = circumference - (probability * circumference);

  return (
    <div className="flex flex-col items-center justify-center">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="transform -rotate-90">
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke="#f3f4f6"
            strokeWidth="12"
            fill="none"
          />
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke={color}
            strokeWidth="12"
            fill="none"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            className="transition-all duration-1000 ease-out"
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-4xl font-bold text-gray-800">{percentage}%</span>
          <span className="text-sm text-gray-500 font-medium mt-1">Probability</span>
        </div>
      </div>
      <div 
        className="mt-4 px-4 py-1.5 rounded-full text-sm font-bold border"
        style={{ color: color, borderColor: color, backgroundColor: `${color}15` }}
      >
        {category}
      </div>
    </div>
  );
};
