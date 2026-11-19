import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight, Activity, Network, Lightbulb, CheckCircle2 } from 'lucide-react';

const Home: React.FC = () => {
  return (
    <div className="space-y-12 pb-12">
      {/* Hero Section */}
      <section className="bg-white rounded-2xl shadow-sm border border-gray-100 p-12 text-center">
        <h1 className="text-4xl md:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-teal-500 mb-6">
          FedGraph-Diabetes
        </h1>
        <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
          Privacy-Aware Explainable Federated Graph Learning for Diabetes Risk Prediction
        </p>
        <div className="text-gray-600 mb-10 max-w-4xl mx-auto leading-relaxed">
          This system improves upon traditional centralized ML models by representing patients as a similarity graph. 
          Using GraphSAGE, we capture relational information between similar patients. Furthermore, to preserve 
          privacy across multiple healthcare institutions, we employ Federated Learning (FedAvg/FedProx) so that 
          patient data never leaves the local hospital server.
        </div>
        <div className="flex justify-center space-x-4">
          <Link to="/predict" className="btn-primary flex items-center">
            Try Prediction <ArrowRight className="ml-2 w-5 h-5" />
          </Link>
          <Link to="/dashboard" className="btn-secondary">
            View Dashboard
          </Link>
        </div>
      </section>

      {/* Research Story Flow */}
      <section>
        <h2 className="section-title text-center mb-8">Research Evolution</h2>
        <div className="flex flex-col md:flex-row items-center justify-center gap-6">
          <div className="card w-full md:w-1/3 text-center">
            <div className="bg-gray-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="font-bold text-gray-500">1</span>
            </div>
            <h3 className="font-bold text-lg mb-2">Base Paper</h3>
            <p className="text-sm text-gray-600">Centralized ML on tabular data with SHAP explainability. Good baseline, but treats patients independently.</p>
          </div>
          <ArrowRight className="hidden md:block w-8 h-8 text-gray-400" />
          <div className="card w-full md:w-1/3 text-center bg-red-50 border-red-100">
            <div className="bg-red-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="font-bold text-red-500">2</span>
            </div>
            <h3 className="font-bold text-lg mb-2 text-red-800">Research Gap</h3>
            <p className="text-sm text-red-700">Ignores latent patient relationships. Centralized learning poses privacy risks for real hospital deployment.</p>
          </div>
          <ArrowRight className="hidden md:block w-8 h-8 text-gray-400" />
          <div className="card w-full md:w-1/3 text-center bg-blue-50 border-blue-100">
            <div className="bg-blue-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
              <span className="font-bold text-blue-600">3</span>
            </div>
            <h3 className="font-bold text-lg mb-2 text-blue-900">Our Solution</h3>
            <p className="text-sm text-blue-800">k-NN Patient Graph + GraphSAGE for relational learning + Federated Learning for privacy preservation.</p>
          </div>
        </div>
      </section>

      {/* Key Technologies */}
      <section>
        <h2 className="section-title text-center mb-8">Key Technologies</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[
            { icon: Activity, title: 'Patient Similarity Graph', desc: 'Connecting patients based on clinical feature similarity using k-Nearest Neighbors.' },
            { icon: Network, title: 'GraphSAGE', desc: 'Inductive representation learning on large graphs, capable of generating embeddings for unseen patients.' },
            { icon: Network, title: 'Federated Learning', desc: 'Training decentralized models across simulated hospitals without exchanging raw data.' },
            { icon: CheckCircle2, title: 'FedAvg', desc: 'Standard federated averaging algorithm for aggregating model weights globally.' },
            { icon: CheckCircle2, title: 'FedProx', desc: 'Advanced aggregation handling non-IID data distribution across heterogeneous hospitals.' },
            { icon: Lightbulb, title: 'Explainable AI', desc: 'Providing transparency into predictions using SHAP and gradient-based saliency methods.' },
          ].map((tech, idx) => (
            <div key={idx} className="card hover:shadow-md transition-shadow">
              <tech.icon className="w-8 h-8 text-blue-500 mb-4" />
              <h3 className="font-bold text-lg mb-2">{tech.title}</h3>
              <p className="text-gray-600 text-sm">{tech.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Disclaimer */}
      <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 rounded-r-lg">
        <div className="flex">
          <div className="ml-3">
            <h3 className="text-sm font-medium text-yellow-800">Medical Disclaimer</h3>
            <div className="mt-2 text-sm text-yellow-700">
              <p>This system is a research prototype for academic purposes only. It is not intended for medical diagnosis, treatment, or clinical use.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;
