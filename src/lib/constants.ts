import { browser, dev } from '$app/environment';
// import { version } from '../../package.json';

export const APP_NAME = 'Open WebUI';

export const WEBUI_HOSTNAME = browser ? (dev ? `${location.hostname}:8080` : ``) : '';
export const WEBUI_BASE_URL = browser ? (dev ? `http://${WEBUI_HOSTNAME}` : ``) : ``;
export const WEBUI_API_BASE_URL = `${WEBUI_BASE_URL}/api/v1`;

export const OLLAMA_API_BASE_URL = `${WEBUI_BASE_URL}/ollama`;
export const OPENAI_API_BASE_URL = `${WEBUI_BASE_URL}/openai`;
export const AUDIO_API_BASE_URL = `${WEBUI_BASE_URL}/api/v1/audio`;
export const IMAGES_API_BASE_URL = `${WEBUI_BASE_URL}/api/v1/images`;
export const RETRIEVAL_API_BASE_URL = `${WEBUI_BASE_URL}/api/v1/retrieval`;

export const WEBUI_VERSION = APP_VERSION;
export const WEBUI_BUILD_HASH = APP_BUILD_HASH;
export const REQUIRED_OLLAMA_VERSION = '0.1.16';

export const SUPPORTED_FILE_TYPE = [
	'application/epub+zip',
	'application/pdf',
	'text/plain',
	'text/csv',
	'text/xml',
	'text/html',
	'text/x-python',
	'text/css',
	'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
	'application/octet-stream',
	'application/x-javascript',
	'text/markdown',
	'audio/mpeg',
	'audio/wav',
	'audio/ogg',
	'audio/x-m4a'
];

export const SUPPORTED_FILE_EXTENSIONS = [
	'md',
	'rst',
	'go',
	'py',
	'java',
	'sh',
	'bat',
	'ps1',
	'cmd',
	'js',
	'ts',
	'css',
	'cpp',
	'hpp',
	'h',
	'c',
	'cs',
	'htm',
	'html',
	'sql',
	'log',
	'ini',
	'pl',
	'pm',
	'r',
	'dart',
	'dockerfile',
	'env',
	'php',
	'hs',
	'hsc',
	'lua',
	'nginxconf',
	'conf',
	'm',
	'mm',
	'plsql',
	'perl',
	'rb',
	'rs',
	'db2',
	'scala',
	'bash',
	'swift',
	'vue',
	'svelte',
	'doc',
	'docx',
	'pdf',
	'csv',
	'txt',
	'xls',
	'xlsx',
	'pptx',
	'ppt',
	'msg'
];

export const PASTED_TEXT_CHARACTER_LIMIT = 1000;


export const FIRST_VISIT_MODAL = {
  ENABLED: true,
  TITLE: "Bienvenue sur LEX Assistant",
  MESSAGE: `
**Important : Version de test**

Bonjour, bienvenue sur la version test d’Apertus !  

Apertus a été utilisé ici pour tous vous aider à accéder rapidement aux informations du Polylex et à les traiter comme vous le souhaitez. Nous vous encourageons à tester l’outils pour nous aider à l’améliorer afin qu’il puisse répondre à vos besoins.  Veuillez noter :  

- Vos requêtes sont anonymes, nous ne gardons pas de trace de celles-ci, ni de vos interactions avec l’outil.  

- Vous pouvez lui écrire dans la langue que vous souhaitez. Ceci est même encouragé !  

- L’outil possède des fonctionnalités de traitement de texte telles traduire, résumer, synthétiser, vulgariser, etc.  

- L’outil reste limité à Polylex pour le moment. Il n’y a pas non plus d’accès au web 

 

Vos retours sont essentiels pour améliorer l’outil : envoyez vos impressions, et si pertinent, des captures d’écran ou exemples de prompts, à : feedback_Apertus@epfl.ch  
  `.trim(),
  BUTTON_TEXT: "J'ai compris"
} as const;

export const RAG_CONFIG = {
  // Forcer le RAG Search activé (non désactivable par l'utilisateur)
  FORCE_RAG_ENABLED: true,
  
  // Cacher le bouton RAG Search de l'interface
  SHOW_RAG_BUTTON: false,
  
  // Bibliothèque par défaut (LEXs)
  DEFAULT_LIBRARY_ID: 'LEX_FR',  // ⚠️ Remplacer par l'ID exact
  
  // Permettre à l'utilisateur de changer de bibliothèque
  ALLOW_LIBRARY_CHANGE: false,
  
  // Mode expert par défaut
  DEFAULT_EXPERT_MODE: false,
} as const;

// Source: https://kit.svelte.dev/docs/modules#$env-static-public
// This feature, akin to $env/static/private, exclusively incorporates environment variables
// that are prefixed with config.kit.env.publicPrefix (usually set to PUBLIC_).
// Consequently, these variables can be securely exposed to client-side code.

