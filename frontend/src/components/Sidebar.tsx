import React from 'react';
import { NavLink } from 'react-router-dom';
import { 
  Home, 
  LayoutDashboard, 
  Database, 
  BarChart2, 
  Network, 
  Activity, 
  Stethoscope, 
  Lightbulb, 
  Info,
  Dna
} from 'lucide-react';

export const Sidebar: React.FC = () => {
  const navItems = [
    { path: '/', label: 'Home', icon: Home },
    { path: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { path: '/dataset', label: 'Dataset Analysis', icon: Database },
    { path: '/models', label: 'Model Comparison', icon: BarChart2 },
    { path: '/federated', label: 'Federated Learning', icon: Network },
    { path: '/graph', label: 'Patient Graph', icon: Activity },
    { path: '/predict', label: 'Prediction', icon: Stethoscope },
    { path: '/explainability', label: 'Explainability', icon: Lightbulb },
    { path: '/about', label: 'About/Methodology', icon: Info },
  ];

  return (
    <aside className="w-64 bg-white border-r border-gray-200 flex flex-col hidden md:flex">
      <div className="h-16 flex items-center px-6 border-b border-gray-200">
        <Dna className="w-8 h-8 text-blue-600 mr-2" />
        <span className="text-lg font-bold text-gray-900 truncate">FedGraph-Diabetes</span>
      </div>
      <div className="flex-1 overflow-y-auto py-4">
        <nav className="space-y-1 px-3">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.path}
                to={item.path}
                className={({ isActive }) =>
                  `flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors ${
                    isActive
                      ? 'bg-blue-50 text-blue-700'
                      : 'text-gray-700 hover:bg-gray-50 hover:text-gray-900'
                  }`
                }
              >
                <Icon className="w-5 h-5 mr-3 flex-shrink-0" />
                {item.label}
              </NavLink>
            );
          })}
        </nav>
      </div>
      <div className="p-4 border-t border-gray-200 text-xs text-gray-500 text-center">
        <a href="#" className="hover:text-gray-700 underline">GitHub Repository</a>
      </div>
    </aside>
  );
};
