import React, { useEffect, useState } from 'react';
import { apiService, FederatedStatus, FederatedHistory } from '../services/api';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { EmptyState } from '../components/EmptyState';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const FederatedLearning: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [status, setStatus] = useState<FederatedStatus | null>(null);
  const [history, setHistory] = useState<FederatedHistory | null>(null);

  useEffect(() => {
    Promise.all([
      apiService.getFederatedStatus(),
      apiService.getFederatedHistory()
    ]).then(([s, h]) => {
      setStatus(s);
      setHistory(h);
      setLoading(false);
    });
  }, []);

  if (loading) return <LoadingSpinner />;
  if (!status?.trained) return <EmptyState title="Federated Learning Not Started" message="Train the federated models on the backend to view learning curves and client statistics." />;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="card">
          <h2 className="section-title">FL Architecture</h2>
          <div className="flex flex-col items-center py-6">
            <div className="bg-blue-100 border-2 border-blue-500 rounded-lg p-4 font-bold text-center w-48 mb-6">
              Global Server
              <div className="text-xs font-normal text-blue-700 mt-1">Aggregates weights</div>
            </div>
            
            <div className="flex justify-center space-x-2 md:space-x-4 w-full">
              {[1, 2, 3, 4, 5].map((client) => (
                <div key={client} className="flex flex-col items-center">
                  <div className="h-10 w-0.5 bg-gray-300 mb-2 border-l-2 border-dashed border-gray-400"></div>
                  <div className="bg-gray-100 border border-gray-300 rounded p-2 text-xs text-center w-16 md:w-20">
                    Hospital {client}
                    <div className="text-[10px] text-gray-500 mt-1">Local Data</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="card">
          <h2 className="section-title">Client Data Distribution</h2>
          <p className="text-sm text-gray-600 mb-4">Simulated non-IID data distribution across {status.num_clients} healthcare institutions.</p>
          <div className="space-y-3">
            {Object.entries(status.clients_data_size).map(([client, size]) => (
              <div key={client}>
                <div className="flex justify-between text-xs mb-1">
                  <span className="font-medium text-gray-700">{client.replace('_', ' ').toUpperCase()}</span>
                  <span className="text-gray-500">{size.toLocaleString()} records</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div className="bg-teal-500 h-2 rounded-full" style={{ width: `${(size / 20000) * 100}%` }}></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="card">
        <h2 className="section-title">Federated Training History (Global Model)</h2>
        <div className="h-80 mt-6">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history?.rounds || []} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="round" label={{ value: 'Communication Round', position: 'insideBottomRight', offset: -10 }} />
              <YAxis yAxisId="left" domain={[0.4, 1.0]} label={{ value: 'Accuracy', angle: -90, position: 'insideLeft' }} />
              <YAxis yAxisId="right" orientation="right" domain={[0, 1.5]} label={{ value: 'Loss', angle: 90, position: 'insideRight' }} />
              <Tooltip />
              <Legend verticalAlign="top" />
              <Line yAxisId="left" type="monotone" dataKey="accuracy" name="Global Accuracy" stroke="#3b82f6" strokeWidth={2} dot={false} />
              <Line yAxisId="right" type="monotone" dataKey="loss" name="Global Loss" stroke="#ef4444" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

export default FederatedLearning;
