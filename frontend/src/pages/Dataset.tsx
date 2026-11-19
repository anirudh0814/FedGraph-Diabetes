import React, { useEffect, useState } from 'react';
import { apiService, DatasetSummary, FeatureInfo } from '../services/api';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const Dataset: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [features, setFeatures] = useState<FeatureInfo[]>([]);

  useEffect(() => {
    const fetchDataset = async () => {
      try {
        const [ds, fs] = await Promise.all([
          apiService.getDatasetSummary(),
          apiService.getDatasetFeatures()
        ]);
        setSummary(ds);
        setFeatures(fs);
      } catch (e) {
        console.error(e);
      } finally {
        setLoading(false);
      }
    };
    fetchDataset();
  }, []);

  if (loading) return <LoadingSpinner text="Loading dataset information..." />;
  if (!summary) return <div>Failed to load dataset data.</div>;

  const getNonDiabetic = () => {
    const d = summary.class_distribution;
    return d['0'] || d[0 as any] || 35346;
  };
  const getDiabetic = () => {
    const d = summary.class_distribution;
    return d['1'] || d[1 as any] || 35346;
  };

  const classData = [
    { name: 'Non-Diabetic (0)', count: getNonDiabetic(), color: '#3b82f6' },
    { name: 'Diabetic (1)',     count: getDiabetic(),    color: '#ef4444' },
  ];

  return (
    <div className="space-y-6">
      <div className="card">
        <h2 className="section-title">Dataset Overview</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="bg-gray-50 p-4 rounded-lg">
            <div className="text-sm text-gray-500">Source</div>
            <div className="font-bold text-lg">{summary.source}</div>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <div className="text-sm text-gray-500">Total Records</div>
            <div className="font-bold text-lg">{summary.total_records.toLocaleString()}</div>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <div className="text-sm text-gray-500">Features</div>
            <div className="font-bold text-lg">{summary.num_features}</div>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <div className="text-sm text-gray-500">Balance</div>
            <div className="font-bold text-lg text-green-600">Balanced (50/50)</div>
          </div>
        </div>

        <div className="mt-8">
          <h3 className="font-bold text-gray-800 mb-4">Class Distribution</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={classData} layout="vertical" margin={{ top: 5, right: 30, left: 100, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                <XAxis type="number" />
                <YAxis dataKey="name" type="category" />
                <Tooltip cursor={{fill: 'transparent'}} />
                <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                  {classData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="card">
        <h2 className="section-title">Feature Dictionary ({features.length} Features)</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Feature</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Type</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Description</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Range/Values</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {features.map((f, i) => (
                <tr key={i} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{f.name}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    <span className={`badge ${
                      f.type === 'binary' ? 'bg-purple-100 text-purple-800' :
                      f.type === 'numeric' ? 'bg-blue-100 text-blue-800' : 'bg-orange-100 text-orange-800'
                    }`}>
                      {f.type}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">{f.description}</td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{f.min} - {f.max}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default Dataset;
