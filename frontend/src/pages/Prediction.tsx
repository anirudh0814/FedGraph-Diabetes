import React, { useState, useEffect } from 'react';
import { apiService, PredictionResult } from '../services/api';
import { RiskGauge } from '../components/RiskGauge';
import { Toast } from '../components/Toast';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, CartesianGrid } from 'recharts';

// All visible form features
const FORM_FEATURES = [
  'HighBP', 'HighChol', 'BMI', 'Smoker', 'Stroke',
  'HeartDiseaseorAttack', 'PhysActivity', 'GenHlth',
  'Age', 'PhysHlth', 'MentHlth', 'DiffWalk',
  'Sex', 'Fruits', 'HvyAlcoholConsump'
];

const FEATURE_LABELS: Record<string, string> = {
  HighBP: 'High Blood Pressure',
  HighChol: 'High Cholesterol',
  BMI: 'BMI',
  Smoker: 'Smoker',
  Stroke: 'History of Stroke',
  HeartDiseaseorAttack: 'Heart Disease/Attack',
  PhysActivity: 'Physical Activity',
  GenHlth: 'General Health (1-5)',
  Age: 'Age Bracket (1-13)',
  PhysHlth: 'Physical Health Days',
  MentHlth: 'Mental Health Days',
  DiffWalk: 'Difficulty Walking',
  Sex: 'Sex',
  Fruits: 'Fruits Daily',
  HvyAlcoholConsump: 'Heavy Alcohol',
};

interface PatientHistory {
  id: number;
  timestamp: string;
  inputs: Record<string, number>;
  probability: number;
  risk_category: string;
  model_used: string;
}

const HISTORY_KEY = 'fedgraph_patient_history';

const getRiskColor = (risk: string) => {
  if (risk === 'High Risk') return 'text-red-600 bg-red-50 border-red-200';
  if (risk === 'Moderate Risk') return 'text-orange-600 bg-orange-50 border-orange-200';
  return 'text-green-600 bg-green-50 border-green-200';
};

const Prediction: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState('');
  const [history, setHistory] = useState<PatientHistory[]>([]);

  // Load history from localStorage on mount
  useEffect(() => {
    const stored = localStorage.getItem(HISTORY_KEY);
    if (stored) {
      try { setHistory(JSON.parse(stored)); } catch {}
    }
  }, []);

  const [formData, setFormData] = useState<Record<string, number>>({
    HighBP: 0, HighChol: 0, BMI: 25, Smoker: 0, Stroke: 0,
    HeartDiseaseorAttack: 0, PhysActivity: 1, Fruits: 1, Veggies: 1,
    HvyAlcoholConsump: 0, AnyHealthcare: 1, NoDocbcCost: 0,
    GenHlth: 3, MentHlth: 0, PhysHlth: 0, DiffWalk: 0,
    Sex: 0, Age: 5, Education: 4, Income: 5
  });

  const handleChange = (e: React.ChangeEvent<HTMLSelectElement | HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: Number(value) }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    try {
      const res = await apiService.predict(formData);
      setResult(res);

      // Save to history (keep last 3)
      const newEntry: PatientHistory = {
        id: Date.now(),
        timestamp: new Date().toLocaleString('en-IN', { hour12: true }),
        inputs: { ...formData },
        probability: res.probability,
        risk_category: res.risk_category || 'Unknown',
        model_used: res.model_used,
      };
      setHistory(prev => {
        const updated = [newEntry, ...prev].slice(0, 3);
        localStorage.setItem(HISTORY_KEY, JSON.stringify(updated));
        return updated;
      });
    } catch (err) {
      setError('Prediction failed. Ensure backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const clearHistory = () => {
    setHistory([]);
    localStorage.removeItem(HISTORY_KEY);
  };

  const renderSelect = (name: string, label: string, options: { val: number; label: string }[]) => (
    <div className="mb-4">
      <label className="label">{label}</label>
      <select name={name} value={formData[name]} onChange={handleChange} className="select">
        {options.map(opt => <option key={opt.val} value={opt.val}>{opt.label}</option>)}
      </select>
    </div>
  );

  const renderInput = (name: string, label: string, min: number, max: number) => (
    <div className="mb-4">
      <label className="label">{label} ({min}-{max})</label>
      <input type="number" name={name} value={formData[name]} onChange={handleChange}
        min={min} max={max} className="input" required />
    </div>
  );

  const binaryOptions = [{ val: 0, label: 'No' }, { val: 1, label: 'Yes' }];

  // Only the form-visible features for the chart
  const inputChartData = FORM_FEATURES.map(key => ({
    feature: FEATURE_LABELS[key] || key,
    value: formData[key],
    key,
  }));

  return (
    <div className="space-y-6">
      <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 rounded text-sm text-yellow-800">
        <strong>Disclaimer:</strong> This tool is for research demonstration only. It does not provide medical advice.
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* ── INPUT FORM ── */}
        <div className="card">
          <h2 className="section-title">Patient Profile Input</h2>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {renderSelect('HighBP', 'High Blood Pressure', binaryOptions)}
              {renderSelect('HighChol', 'High Cholesterol', binaryOptions)}
              {renderInput('BMI', 'BMI', 10, 90)}
              {renderSelect('Smoker', 'Smoker (>100 lifetime)', binaryOptions)}
              {renderSelect('Stroke', 'History of Stroke', binaryOptions)}
              {renderSelect('HeartDiseaseorAttack', 'Heart Disease / Attack', binaryOptions)}
              {renderSelect('PhysActivity', 'Physical Activity (30 days)', binaryOptions)}
              {renderSelect('GenHlth', 'General Health', [
                { val: 1, label: 'Excellent' }, { val: 2, label: 'Very Good' },
                { val: 3, label: 'Good' }, { val: 4, label: 'Fair' }, { val: 5, label: 'Poor' }
              ])}
              {renderSelect('Age', 'Age Bracket', [
                { val: 1, label: '18-24' }, { val: 3, label: '30-34' }, { val: 5, label: '40-44' },
                { val: 7, label: '50-54' }, { val: 9, label: '60-64' }, { val: 11, label: '70-74' },
                { val: 13, label: '80+' }
              ])}
              {renderInput('PhysHlth', 'Poor Physical Health Days (0-30)', 0, 30)}
              {renderInput('MentHlth', 'Poor Mental Health Days (0-30)', 0, 30)}
              {renderSelect('DiffWalk', 'Difficulty Walking', binaryOptions)}
              {renderSelect('Sex', 'Sex', [{ val: 0, label: 'Female' }, { val: 1, label: 'Male' }])}
              {renderSelect('Fruits', 'Consume Fruits Daily', binaryOptions)}
              {renderSelect('HvyAlcoholConsump', 'Heavy Alcohol Consumption', binaryOptions)}
            </div>
            <button type="submit" className="btn-primary w-full mt-6 flex justify-center" disabled={loading}>
              {loading ? 'Analyzing...' : 'Predict Diabetes Risk'}
            </button>
          </form>
        </div>

        {/* ── RESULTS ── */}
        <div>
          {result ? (
            <div className="space-y-6">
              {/* Risk gauge */}
              <div className="card flex flex-col items-center justify-center py-8">
                <h2 className="text-xl font-bold mb-6 text-gray-800">Prediction Result</h2>
                <RiskGauge probability={result.probability} size={240} />
                <div className="mt-6 text-sm text-gray-500 bg-gray-50 px-4 py-2 rounded">
                  Model Used: <strong>{result.model_used}</strong>
                </div>
              </div>

              {/* Patient Input Values — only form features */}
              <div className="card">
                <h3 className="font-bold text-gray-800 mb-1">Patient Input Values</h3>
                <p className="text-xs text-gray-500 mb-3">Values you entered (12 clinical features).</p>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart layout="vertical" data={inputChartData}
                      margin={{ top: 0, right: 30, left: 140, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                      <XAxis type="number" />
                      <YAxis dataKey="feature" type="category" tick={{ fontSize: 10 }} width={135} />
                      <Tooltip formatter={(v: any) => [v, 'Input Value']} />
                      <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                        {inputChartData.map(({ key, value }, index) => {
                          const riskKeys = ['HighBP', 'HighChol', 'Stroke', 'HeartDiseaseorAttack', 'DiffWalk'];
                          const fill =
                            riskKeys.includes(key) && value === 1 ? '#ef4444'
                            : key === 'GenHlth' && value >= 4 ? '#ef4444'
                            : key === 'Age' && value >= 10 ? '#f97316'
                            : key === 'BMI' && value >= 30 ? '#f97316'
                            : key === 'MentHlth' && value >= 15 ? '#f97316'
                            : key === 'PhysHlth' && value >= 15 ? '#f97316'
                            : '#3b82f6';
                          return <Cell key={`cell-${index}`} fill={fill} />;
                        })}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="flex gap-4 mt-2 justify-center text-xs">
                  <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-red-500 inline-block" /> High Risk</span>
                  <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-orange-400 inline-block" /> Elevated</span>
                  <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-blue-500 inline-block" /> Normal</span>
                </div>
              </div>

              {/* Risk Factors — Grouped bars: red=risk, green=protection, both shown for every feature */}
              <div className="card">
                <h3 className="font-bold text-gray-800 mb-1">Patient-Specific Risk Factors</h3>
                <p className="text-xs text-gray-500 mb-3">
                  Every feature shows both: 🔴 Risk level (red) and 🟢 Protection level (green). Longer bar = stronger effect.
                </p>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      layout="vertical"
                      data={[...result.explanation.contributions]
                        .filter(c => FORM_FEATURES.includes(c.feature))
                        .sort((a, b) => Math.abs(b.importance) - Math.abs(a.importance))
                        .map(c => ({
                          feature: c.feature,
                          // Red bar: how much this feature is ADDING risk (positive contribution)
                          risk:       c.importance > 0 ? parseFloat(c.importance.toFixed(3)) : 0.01,
                          // Green bar: how much this feature is REDUCING risk (negative contribution → shown as positive)
                          protection: c.importance < 0 ? parseFloat(Math.abs(c.importance).toFixed(3)) : 0.01,
                        }))}
                      margin={{ top: 0, right: 20, left: 140, bottom: 0 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} />
                      <XAxis type="number" />
                      <YAxis dataKey="feature" type="category" tick={{ fontSize: 10 }} width={135} />
                      <Tooltip
                        formatter={(value: any, name: string) => [
                          Number(value) === 0.01 ? '—' : Number(value).toFixed(3),
                          name === 'risk' ? '🔴 Risk Score' : '🟢 Protection Score'
                        ]}
                      />
                      <Bar dataKey="risk"       name="risk"       fill="#ef4444" radius={[0, 3, 3, 0]} barSize={8} />
                      <Bar dataKey="protection" name="protection" fill="#22c55e" radius={[0, 3, 3, 0]} barSize={8} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="flex gap-6 mt-2 justify-center text-xs">
                  <span className="flex items-center gap-1"><span className="w-4 h-3 rounded bg-red-500 inline-block"/> 🔴 Risk Score (longer = more risky)</span>
                  <span className="flex items-center gap-1"><span className="w-4 h-3 rounded bg-green-500 inline-block"/> 🟢 Protection Score (longer = more protective)</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="card h-64 flex flex-col items-center justify-center text-gray-400 bg-gray-50 border-dashed">
              <p>Fill out the patient profile and click predict to see results.</p>
            </div>
          )}
        </div>
      </div>

      {/* ── PATIENT HISTORY (last 3) ── */}
      {history.length > 0 && (
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="font-bold text-gray-800 text-lg">Recent Patient History (Last 3)</h2>
            <button onClick={clearHistory}
              className="text-xs text-red-500 hover:text-red-700 border border-red-200 px-3 py-1 rounded-lg hover:bg-red-50 transition-colors">
              Clear History
            </button>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {history.map((h, i) => (
              <div key={h.id} className="border border-gray-200 rounded-xl p-4 bg-gray-50 hover:shadow-md transition-shadow">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-semibold text-gray-500">Patient #{history.length - i}</span>
                  <span className={`text-xs font-bold px-2 py-0.5 rounded-full border ${getRiskColor(h.risk_category)}`}>
                    {h.risk_category}
                  </span>
                </div>
                <div className="text-2xl font-bold text-gray-800 mb-1">
                  {(h.probability * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-gray-400 mb-3">{h.timestamp}</div>
                <div className="space-y-1">
                  {FORM_FEATURES.slice(0, 6).map(key => (
                    <div key={key} className="flex justify-between text-xs">
                      <span className="text-gray-500 truncate mr-2">{FEATURE_LABELS[key]}</span>
                      <span className={`font-semibold ${
                        ['HighBP','HighChol','Stroke','HeartDiseaseorAttack','DiffWalk'].includes(key) && h.inputs[key] === 1
                          ? 'text-red-600' : 'text-gray-700'
                      }`}>
                        {key === 'GenHlth'
                          ? ['','Excellent','Very Good','Good','Fair','Poor'][h.inputs[key]] || h.inputs[key]
                          : key === 'Age'
                          ? ['','18-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-74','75-79','80+'][h.inputs[key]] || h.inputs[key]
                          : typeof h.inputs[key] === 'number' && h.inputs[key] <= 1 && ['HighBP','HighChol','Smoker','Stroke','HeartDiseaseorAttack','PhysActivity','DiffWalk'].includes(key)
                          ? (h.inputs[key] === 1 ? 'Yes' : 'No')
                          : h.inputs[key]}
                      </span>
                    </div>
                  ))}
                </div>
                <div className="mt-2 pt-2 border-t border-gray-200 text-xs text-gray-400">
                  Model: {h.model_used}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {error && <Toast message={error} type="error" onClose={() => setError('')} />}
    </div>
  );
};

export default Prediction;
