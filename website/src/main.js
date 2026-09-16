import './style.css';
import { initSpecterScene } from './scene.js';

const WHEEL = 'releases/specter_decision_engine-0.2.0rc2-py3-none-any.whl';
const SHA256 = '6b2274ba286e5ef872ba241422524977ccab5c340b8567fc261ce1c283cff988';
const wheelUrl = new URL(WHEEL, window.location.href).href;
const llmUrl = new URL('llms-install.txt', window.location.href).href;

const commands = {
  windows: [
    'py -m venv .venv',
    `.\\.venv\\Scripts\\python -m pip install "${wheelUrl}"`,
    '.\\.venv\\Scripts\\python -m specter_decision doctor',
  ].join('\n'),
  unix: [
    'python3 -m venv .venv',
    `./.venv/bin/python -m pip install "${wheelUrl}"`,
    './.venv/bin/python -m specter_decision doctor',
  ].join('\n'),
};

let selectedOS = /Win/i.test(navigator.userAgent) ? 'windows' : 'unix';
const commandEl = document.querySelector('#install-command');
const llmUrlEl = document.querySelector('#llm-url');
const tabs = [...document.querySelectorAll('.os-tab')];

function renderInstall() {
  commandEl.textContent = commands[selectedOS];
  tabs.forEach((tab) => {
    const active = tab.dataset.os === selectedOS;
    tab.classList.toggle('active', active);
    tab.setAttribute('aria-selected', String(active));
  });
}

renderInstall();
llmUrlEl.textContent = llmUrl;
tabs.forEach((tab) => tab.addEventListener('click', () => {
  selectedOS = tab.dataset.os;
  renderInstall();
}));

const llmPrompt = `Install Specter Decision Engine 0.2.0rc2 for me using the official machine-readable instructions at:
${llmUrl}

Treat that URL as the source of truth. Use an isolated virtual environment, verify Python >= 3.11, verify the exact wheel SHA-256 (${SHA256}), install only the pinned artifact, and run the package doctor check. Do not substitute a similarly named PyPI package or claim that a trained model/calibration is bundled. Explain any blocker before changing course.`;

const toast = document.querySelector('.toast');
let toastTimer;
async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    toast.textContent = 'Copied to clipboard';
  } catch {
    toast.textContent = 'Copy blocked by browser';
  }
  toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 1600);
}

document.querySelectorAll('[data-copy]').forEach((button) => {
  button.addEventListener('click', () => {
    const kind = button.dataset.copy;
    if (kind === 'install') copyText(commands[selectedOS]);
    if (kind === 'llm') copyText(llmPrompt);
    if (kind === 'url') copyText(llmUrl);
  });
});
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      entry.target.classList.add('visible');
      observer.unobserve(entry.target);
    }
  });
}, { threshold: 0.12, rootMargin: '0px 0px -40px' });

document.querySelectorAll('.reveal').forEach((element) => {
  if (reducedMotion) element.classList.add('visible');
  else observer.observe(element);
});

if (!reducedMotion) initSpecterScene(document.querySelector('#specter-canvas'));
else document.querySelector('#specter-canvas').hidden = true;

window.addEventListener('scroll', () => {
  const topbar = document.querySelector('.topbar');
  topbar.style.background = window.scrollY > 36
    ? 'rgba(5, 8, 7, .84)'
    : 'rgba(5, 8, 7, .65)';
}, { passive: true });
