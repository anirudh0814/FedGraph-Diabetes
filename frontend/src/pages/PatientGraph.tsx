import React, { useEffect, useState } from 'react';
import { apiService, GraphStatistics, SampleGraph } from '../services/api';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { MetricCard } from '../components/MetricCard';
import { Network, Users, Share2, AlertCircle } from 'lucide-react';

const PatientGraph: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState<GraphStatistics | null>(null);
  const [graphData, setGraphData] = useState<SampleGraph | null>(null);

  useEffect(() => {
    Promise.all([
      apiService.getGraphStatistics(),
      apiService.getSampleGraph()
    ]).then(([s, g]) => {
      setStats(s);
      setGraphData(g);
      setLoading(false);
    });
  }, []);

  if (loading) return <LoadingSpinner />;

  // Simple force-directed layout approximation for SVG
  const width = 600;
  const height = 400;
  
  const nodesWithPos = graphData?.nodes.map((n, i) => {
    // Distribute randomly but somewhat centrally
    const r = Math.random() * (height / 2.5);
    const theta = Math.random() * 2 * Math.PI;
    return {
      ...n,
      x: width/2 + r * Math.cos(theta),
      y: height/2 + r * Math.sin(theta)
    };
  }) || [];

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard title="Total Nodes" value={stats?.num_nodes.toLocaleString() || 0} icon={<Users className="w-5 h-5"/>} subtitle="Patients" />
        <MetricCard title="Total Edges" value={stats?.num_edges.toLocaleString() || 0} icon={<Network className="w-5 h-5"/>} subtitle="Similarity connections" />
        <MetricCard title="k-Neighbors" value={stats?.k_neighbors || 5} icon={<Share2 className="w-5 h-5"/>} subtitle="Connections per node" />
        <MetricCard title="Density" value={(stats?.density || 0).toFixed(6)} icon={<AlertCircle className="w-5 h-5"/>} subtitle="Graph sparsity" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 card flex flex-col items-center">
          <h2 className="section-title w-full">Sample k-NN Patient Graph (Subgraph)</h2>
          <div className="w-full bg-gray-50 rounded-lg border border-gray-200 flex justify-center items-center overflow-hidden" style={{ minHeight: '400px' }}>
            {nodesWithPos.length > 0 ? (
              <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`}>
                {/* Draw edges */}
                {graphData?.edges.map((edge, i) => {
                  const source = nodesWithPos.find(n => n.id === edge.source);
                  const target = nodesWithPos.find(n => n.id === edge.target);
                  if (!source || !target) return null;
                  return (
                    <line 
                      key={`e${i}`} 
                      x1={source.x} y1={source.y} 
                      x2={target.x} y2={target.y} 
                      stroke="#e5e7eb" 
                      strokeWidth="1" 
                    />
                  );
                })}
                {/* Draw nodes */}
                {nodesWithPos.map((node, i) => (
                  <circle 
                    key={`n${i}`} 
                    cx={node.x} 
                    cy={node.y} 
                    r="4" 
                    fill={node.diabetes === 1 ? '#ef4444' : '#3b82f6'} 
                    stroke="#fff"
                    strokeWidth="1"
                  >
                    <title>Patient {node.id} - {node.diabetes === 1 ? 'Diabetic' : 'Healthy'}</title>
                  </circle>
                ))}
              </svg>
            ) : (
              <div className="text-gray-500">Graph data not available</div>
            )}
          </div>
          <div className="flex gap-4 mt-4 w-full justify-center text-sm">
            <div className="flex items-center"><div className="w-3 h-3 rounded-full bg-blue-500 mr-2"></div> Non-Diabetic</div>
            <div className="flex items-center"><div className="w-3 h-3 rounded-full bg-red-500 mr-2"></div> Diabetic</div>
          </div>
        </div>

        <div className="card space-y-6">
          <div>
            <h3 className="font-bold text-gray-900 mb-2">How it works</h3>
            <p className="text-sm text-gray-600">
              Unlike traditional models that treat patients as isolated rows in a table, we construct a <strong>k-Nearest Neighbor (k-NN) graph</strong>.
            </p>
          </div>
          
          <div className="space-y-4">
            <div className="flex items-start">
              <div className="bg-blue-100 text-blue-800 rounded-full w-6 h-6 flex items-center justify-center text-xs font-bold mr-3 mt-0.5 shrink-0">1</div>
              <p className="text-sm text-gray-600">Extract clinical features (BMI, Age, BP, etc.) for each patient.</p>
            </div>
            <div className="flex items-start">
              <div className="bg-blue-100 text-blue-800 rounded-full w-6 h-6 flex items-center justify-center text-xs font-bold mr-3 mt-0.5 shrink-0">2</div>
              <p className="text-sm text-gray-600">Calculate Euclidean distance between all pairs of patients in the feature space.</p>
            </div>
            <div className="flex items-start">
              <div className="bg-blue-100 text-blue-800 rounded-full w-6 h-6 flex items-center justify-center text-xs font-bold mr-3 mt-0.5 shrink-0">3</div>
              <p className="text-sm text-gray-600">Connect each patient to their <em>k</em> most similar peers with an edge.</p>
            </div>
            <div className="flex items-start">
              <div className="bg-blue-100 text-blue-800 rounded-full w-6 h-6 flex items-center justify-center text-xs font-bold mr-3 mt-0.5 shrink-0">4</div>
              <p className="text-sm text-gray-600">Train GraphSAGE to aggregate information from these neighbors to predict risk.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PatientGraph;
