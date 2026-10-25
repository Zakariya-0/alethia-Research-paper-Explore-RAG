// Exemples calibrés sur le corpus indexé (verdicts attendus : débattu, preuves insuffisantes, soutenu).
export const EXAMPLES = [
  { label: 'Mémoire des world models', claim: 'Video world models retain visual memory of generated content' },
  { label: 'Velox et mémoire spatiale', claim: 'Velox uses spatial memory for video world models' },
  { label: 'Diffusion et images', claim: 'Diffusion models improve image synthesis' },
];

// Bornes imposées par l'API (VerificationRequest).
export const MIN_CLAIM = 5;
export const MAX_CLAIM = 500;
