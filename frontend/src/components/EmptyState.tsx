import React from 'react';
import { AlertCircle } from 'lucide-react';

interface EmptyStateProps {
  title: string;
  message: string;
  actionLabel?: string;
  onAction?: () => void;
}

export const EmptyState: React.FC<EmptyStateProps> = ({ title, message, actionLabel, onAction }) => {
  return (
    <div className="flex flex-col items-center justify-center p-12 bg-white rounded-xl border border-gray-200 border-dashed text-center">
      <div className="bg-gray-100 p-3 rounded-full mb-4">
        <AlertCircle className="w-8 h-8 text-gray-400" />
      </div>
      <h3 className="text-lg font-medium text-gray-900 mb-1">{title}</h3>
      <p className="text-sm text-gray-500 mb-6 max-w-md">{message}</p>
      {actionLabel && onAction && (
        <button onClick={onAction} className="btn-primary">
          {actionLabel}
        </button>
      )}
    </div>
  );
};
