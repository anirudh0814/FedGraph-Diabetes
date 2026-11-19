import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Home from './pages/Home';
import Dashboard from './pages/Dashboard';
import Dataset from './pages/Dataset';
import ModelComparison from './pages/ModelComparison';
import FederatedLearning from './pages/FederatedLearning';
import PatientGraph from './pages/PatientGraph';
import Prediction from './pages/Prediction';
import Explainability from './pages/Explainability';
import About from './pages/About';

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/dataset" element={<Dataset />} />
          <Route path="/models" element={<ModelComparison />} />
          <Route path="/federated" element={<FederatedLearning />} />
          <Route path="/graph" element={<PatientGraph />} />
          <Route path="/predict" element={<Prediction />} />
          <Route path="/explainability" element={<Explainability />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;
