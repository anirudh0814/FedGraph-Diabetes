import React, { useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';
import { apiService } from '../services/api';

export const Header: React.FC = () => {
  const location = useLocation();
  const [isDemo, setIsDemo] = useState(false);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const health = await apiService.getHealth();
        setIsConnected(health.status === 'healthy' || health.status === 'ok');
        setIsDemo(health.demo_mode);
      } catch (e) {
        setIsConnected(false);
      }
    };
    checkHealth();
    // Check every 30s
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const getPageTitle = () => {
    const path = location.pathname;
    if (path === '/') return 'Home';
    if (path === '/dashboard') return 'System Dashboard';
    if (path === '/dataset') return 'Dataset Analysis (BRFSS 2015)';
    if (path === '/models') return 'Model Comparison & Evaluation';
    if (path === '/federated') return 'Federated Learning Insights';
    if (path === '/graph') return 'Patient Similarity Graph (k-NN)';
    if (path === '/predict') return 'Diabetes Risk Prediction';
    if (path === '/explainability') return 'Model Explainability';
    if (path === '/about') return 'About & Methodology';
    return '';
  };

  return (
    <header className="h-16 bg-white border-b border-gray-200 flex items-center justify-between px-6 z-10 shadow-sm">
      <h1 className="text-xl font-semibold text-gray-800">{getPageTitle()}</h1>
      <div className="flex items-center space-x-4">

        <div className="flex items-center space-x-2">
          <div className={`w-2.5 h-2.5 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
          <span className="text-sm text-gray-600 font-medium">
            {isConnected ? 'API Connected' : 'API Offline'}
          </span>
        </div>
      </div>
    </header>
  );
};
