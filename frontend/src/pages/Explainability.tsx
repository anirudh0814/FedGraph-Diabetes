import React, { useEffect, useState } from 'react';
import { apiService, GlobalExplanation } from '../services/api';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { Lightbulb, Info } from 'lucide-react';

const Explainability: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [explanation, setExplanation] = useState<GlobalExplanation | null>(null);

  useEffect(() => {
    apiService.getGlobalExplanation().then(res => {
      setExplanation(res);
      setLoading(false);
    });
  }, []);

  if (loading) return <LoadingSpinner />;

  return (
    <div className="space-y-6">
      <div className="card bg-gradient-to-r from-blue-50 to-teal-50 border-none">
        <div className="flex items-start">
          <Lightbulb className="w-8 h-8 text-blue-600 mr-4 shrink-0" />
          <div>
            <h2 className="text-xl font-bold text-gray-900 mb-2">Explainable AI (XAI) Methodology</h2>
            <p className="text-gray-700 text-sm leading-relaxed">
              Medical professionals require transparent models. We employ two distinct explanation approaches:
              <br/>1. <strong>SHAP (SHapley Additive exPlanations)</strong> for centralized tree-based baseline models.
              <br/>2. <strong>Gradient-based Saliency Mapping</strong> for the GraphSAGE models, calculating the gradient of the output prediction with respect to the input features.
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 card">
          <h2 className="section-title">Global Feature Importance (Dataset Level)</h2>
          <p className="text-sm text-gray-500 mb-6">Calculated using {explanation?.method} on the {explanation?.model} model.</p>
          
          <div className="h-96">
            {explanation && (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart layout="vertical" data={explanation.top_features} margin={{ top: 5, right: 30, left: 100, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} />
                  <XAxis type="number" />
                  <YAxis dataKey="feature" type="category" tick={{fontSize: 12, fontWeight: 500}} />
                  <Tooltip />
                  <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                    {explanation.top_features.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.importance > 0 ? '#ef4444' : '#22c55e'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        <div className="space-y-6">
          <div className="card bg-gray-50">
            <h3 className="font-bold text-gray-900 flex items-center mb-3">
              <Info className="w-4 h-4 mr-2" />
              Clinical Insights
            </h3>
            <ul className="space-y-3 text-sm text-gray-700">
              <li><strong>GenHlth (General Health):</strong> The strongest predictor. Poor general health correlates highly with diabetes risk.</li>
              <li><strong>HighBP & HighChol:</strong> Classical metabolic syndrome indicators, heavily weighted by the model.</li>
              <li><strong>BMI:</strong> Higher Body Mass Index consistently increases predicted risk across all models.</li>
              <li><strong>Income/Education:</strong> Show negative correlation (protective factors). Higher socioeconomic status correlates with lower risk.</li>
            </ul>
          </div>

          <div className="card border-orange-200">
            <h3 className="font-bold text-orange-800 mb-2">Graph vs Tabular Expalnations</h3>
            <p className="text-sm text-gray-600">
              While Tabular models (RF, XGBoost) only consider the patient's own features, the <strong>GraphSAGE model</strong> aggregates features from similar patients in the k-NN graph. Therefore, an explanation for GraphSAGE includes not only the patient's own feature saliency, but also the aggregated neighborhood saliency.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Explainability;
