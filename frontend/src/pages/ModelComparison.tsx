import React, { useEffect, useState } from 'react';
import { apiService, ModelComparison as ModelsType } from '../services/api';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { EmptyState } from '../components/EmptyState';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const ModelComparison: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [models, setModels] = useState<ModelsType | null>(null);

  useEffect(() => {
    apiService.getModelsComparison().then(res => {
      setModels(res);
      setLoading(false);
    });
  }, []);

  if (loading) return <LoadingSpinner />;
  if (!models || Object.keys(models).length === 0) {
    return <EmptyState title="No Models Trained" message="Run the training scripts on the backend to see comparison metrics." />;
  }

  const chartData = Object.entries(models).map(([name, m]) => ({
    name,
    Accuracy: Number((m.accuracy * 100).toFixed(1)),
    F1: Number((m.f1 * 100).toFixed(1)),
    AUC: Number((m.roc_auc * 100).toFixed(1)),
  }));

  return (
    <div className="space-y-6">
      <div className="card">
        <h2 className="section-title">Comprehensive Model Comparison</h2>
        <div className="mb-4 text-sm text-gray-600">
          Comparing Centralized Baseline models (from base paper reproduction) vs Centralized GraphSAGE vs Federated GraphSAGE approaches.
        </div>
        
        <div className="h-96 mt-6">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} tick={{fontSize: 12}} />
              <YAxis domain={[50, 100]} />
              <Tooltip formatter={(val) => [`${val}%`, '']} />
              <Legend verticalAlign="top" height={36}/>
              <Bar dataKey="Accuracy" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              <Bar dataKey="F1" fill="#14b8a6" radius={[4, 4, 0, 0]} />
              <Bar dataKey="AUC" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card">
        <h2 className="section-title">Detailed Metrics Table</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Model</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Type</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Accuracy</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">F1 Score</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">ROC-AUC</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Recall</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {Object.entries(models).map(([name, m], i) => {
                const lowerName = name.toLowerCase();
                const isFed = lowerName.includes('fed');
                const isGraph = lowerName.includes('graphsage') || lowerName.includes('graph');
                
                const typeLabel = isFed 
                  ? 'Federated Graph Learning' 
                  : isGraph 
                  ? 'Centralized Graph Learning (GNN)' 
                  : 'Centralized Tabular Baseline';

                const displayLabel = name
                  .replace('federated_fedavg', 'FedAvg (Federated GraphSAGE)')
                  .replace('federated_fedprox', 'FedProx (Federated GraphSAGE)')
                  .replace('graphsage_centralized', 'GraphSAGE (Centralized GNN)')
                  .replace(/_/g, ' ')
                  .replace(/\b\w/g, l => l.toUpperCase());

                return (
                  <tr key={i} className={`hover:bg-gray-50 ${isGraph || isFed ? 'bg-blue-50/40' : ''}`}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-bold text-gray-900">
                      {displayLabel}
                      {(isGraph || isFed) && (
                        <span className="ml-2 inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold bg-blue-100 text-blue-800">
                          Proposed
                        </span>
                      )}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                        isFed ? 'bg-purple-100 text-purple-800' : isGraph ? 'bg-indigo-100 text-indigo-800' : 'bg-gray-100 text-gray-800'
                      }`}>
                        {typeLabel}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-semibold text-gray-900">{(m.accuracy * 100).toFixed(2)}%</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{(m.f1 * 100).toFixed(2)}%</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{(m.roc_auc * 100).toFixed(2)}%</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{(m.recall * 100).toFixed(2)}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default ModelComparison;
