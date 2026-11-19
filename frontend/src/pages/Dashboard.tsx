import React, { useEffect, useState } from 'react';
import { apiService, DatasetSummary, GraphStatistics, ModelComparison, FederatedStatus } from '../services/api';
import { MetricCard } from '../components/MetricCard';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { Database, Network, Activity, Server, Users, Layers } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer } from 'recharts';
import { Link } from 'react-router-dom';

const Dashboard: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [dataset, setDataset] = useState<DatasetSummary | null>(null);
  const [graph, setGraph] = useState<GraphStatistics | null>(null);
  const [models, setModels] = useState<ModelComparison | null>(null);
  const [fedStatus, setFedStatus] = useState<FederatedStatus | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [ds, gs, ms, fs] = await Promise.all([
          apiService.getDatasetSummary(),
          apiService.getGraphStatistics(),
          apiService.getModelsComparison(),
          apiService.getFederatedStatus()
        ]);
        setDataset(ds);
        setGraph(gs);
        setModels(ms);
        setFedStatus(fs);
      } catch (e) {
        console.error("Error fetching dashboard data", e);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) return <LoadingSpinner text="Loading dashboard metrics..." />;

  const chartData = models ? Object.entries(models).map(([name, metrics]) => ({
    name: name.replace('GraphSAGE_', 'GS_'),
    Accuracy: Number((metrics.accuracy * 100).toFixed(1)),
    AUC: Number((metrics.roc_auc * 100).toFixed(1))
  })) : [];

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold text-gray-900">System Dashboard</h1>
        <div className="space-x-3">
          <Link to="/predict" className="btn-primary text-sm px-4 py-2">Try Prediction</Link>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <MetricCard 
          title="Dataset Records" 
          value={dataset?.total_records.toLocaleString() || 0} 
          icon={<Database className="w-5 h-5" />} 
          subtitle="BRFSS 2015"
        />
        <MetricCard 
          title="Features Used" 
          value={dataset?.num_features || 0} 
          icon={<Layers className="w-5 h-5" />} 
          subtitle="Clinical & Demographic"
        />
        <MetricCard 
          title="Simulated Hospitals" 
          value={fedStatus?.num_clients || 5} 
          icon={<Server className="w-5 h-5" />} 
          subtitle="Federated Clients"
        />
        <MetricCard 
          title="Graph Nodes" 
          value={graph?.num_nodes.toLocaleString() || 0} 
          icon={<Users className="w-5 h-5" />} 
          subtitle="Patients in Graph"
        />
        <MetricCard 
          title="Graph Edges" 
          value={graph?.num_edges.toLocaleString() || 0} 
          icon={<Network className="w-5 h-5" />} 
          subtitle="Similarity Connections"
        />
        <MetricCard 
          title="Avg Node Degree" 
          value={graph?.avg_degree || 0} 
          icon={<Activity className="w-5 h-5" />} 
          subtitle="Neighbors per patient"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-8">
        <div className="card">
          <h2 className="section-title">Model Performance Overview</h2>
          {chartData.length > 0 ? (
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="name" tick={{fontSize: 12}} />
                  <YAxis domain={[50, 100]} />
                  <RechartsTooltip formatter={(val) => [`${val}%`, '']} />
                  <Legend />
                  <Bar dataKey="Accuracy" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="AUC" fill="#14b8a6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="h-80 flex items-center justify-center bg-gray-50 rounded border border-dashed">
              <span className="text-gray-500">Models not trained yet</span>
            </div>
          )}
        </div>

        <div className="card">
          <h2 className="section-title">Pipeline Status</h2>
          <div className="space-y-4 mt-6">
            {[
              { label: 'Dataset Loaded', status: true },
              { label: 'Data Preprocessing', status: true },
              { label: 'Patient Graph Construction', status: !!graph?.num_nodes },
              { label: 'Baseline Models Trained', status: !!(models && Object.keys(models).some(k => ['random_forest','decision_tree','gradient_boosting','svm','logistic_regression'].includes(k))) },
              { label: 'Federated Training', status: fedStatus?.trained || false },
            ].map((step, idx) => (
              <div key={idx} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <span className="font-medium text-gray-700">{step.label}</span>
                <span className={`badge ${step.status ? 'bg-green-100 text-green-800' : 'bg-gray-200 text-gray-600'}`}>
                  {step.status ? 'Completed' : 'Pending'}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
