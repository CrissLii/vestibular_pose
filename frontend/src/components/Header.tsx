import { Activity } from 'lucide-react';
import './Header.css';

export function Header() {
  return (
    <header className="header">
      <div className="header-inner">
        <div className="header-brand">
          <Activity size={28} className="header-icon" />
          <div>
            <h1 className="header-title">儿童统感训练评估系统</h1>
            <p className="header-sub">Vestibular Training Pose Analysis</p>
          </div>
        </div>
      </div>
    </header>
  );
}
