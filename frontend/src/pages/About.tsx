import React from 'react';

const About: React.FC = () => {
  return (
    <div className="max-w-4xl mx-auto space-y-8 pb-12">
      <div className="card bg-blue-600 text-white border-none shadow-md">
        <h1 className="text-3xl font-bold mb-4">About the Research</h1>
        <p className="text-blue-100 text-lg">
          Privacy-Aware Explainable Federated Graph Learning for Diabetes Risk Prediction
        </p>
      </div>

      <div className="card">
        <h2 className="text-xl font-bold text-gray-900 mb-4 border-b pb-2">Research Methodology</h2>
        <div className="prose prose-blue max-w-none text-gray-700 space-y-4">
          <p>
            This project advances the field of healthcare predictive modeling by addressing two critical limitations of standard approaches: 
            <strong> relational isolation</strong> and <strong>data privacy</strong>.
          </p>
          
          <h3 className="text-lg font-semibold mt-6 text-gray-900">1. Moving Beyond Tabular Data</h3>
          <p>
            Standard Machine Learning models (like Random Forest or Logistic Regression) treat every patient as an independent row in a table. 
            However, patients share similarities. By constructing a <strong>k-Nearest Neighbor (k-NN) graph</strong> based on clinical features, 
            we map the dataset into a network where similar patients are connected. We then apply <strong>GraphSAGE (Graph Sample and Aggregate)</strong>, 
            a spatial Graph Neural Network that learns not just from a patient's own features, but by aggregating information from their "neighborhood" of similar patients.
          </p>

          <h3 className="text-lg font-semibold mt-6 text-gray-900">2. Privacy through Federated Learning</h3>
          <p>
            In reality, patient data is siloed across different hospitals and protected by laws like HIPAA. Centralizing data is often legally and technically impossible. 
            We simulate a <strong>Federated Learning (FL)</strong> environment with 5 independent clients (hospitals). 
            Instead of sharing raw patient data, hospitals train the GraphSAGE model locally on their own graph sub-networks and only send mathematical model updates (gradients/weights) 
            to a central server. The server aggregates these using algorithms like <strong>FedAvg</strong> or <strong>FedProx</strong>.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="card">
          <h2 className="text-lg font-bold text-gray-900 mb-4 border-b pb-2">The Dataset</h2>
          <ul className="list-disc pl-5 text-gray-700 text-sm space-y-2">
            <li><strong>Source:</strong> Behavioral Risk Factor Surveillance System (BRFSS) 2015</li>
            <li><strong>Records:</strong> 70,692 perfectly balanced records (35,346 diabetic, 35,346 non-diabetic)</li>
            <li><strong>Features:</strong> 21 clinical and demographic indicators</li>
            <li><strong>Preprocessing:</strong> Standardized numerical features, binary encoding for categorical</li>
          </ul>
        </div>
        
        <div className="card">
          <h2 className="text-lg font-bold text-gray-900 mb-4 border-b pb-2">Limitations & Future Work</h2>
          <ul className="list-disc pl-5 text-gray-700 text-sm space-y-2">
            <li><strong>Simulation:</strong> The federated clients are simulated on a single server, not actual distributed machines.</li>
            <li><strong>Graph Construction:</strong> The k-NN graph is built purely on feature similarity, not real-world relationships (like family ties).</li>
            <li><strong>Compute Costs:</strong> GNNs within Federated settings require significant computational overhead compared to standard models.</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default About;
