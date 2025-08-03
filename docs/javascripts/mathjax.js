// MathJax configuration for Segmentation Robustness Framework documentation

window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: 'tex2jax_ignore',
    processHtmlClass: 'tex2jax_process'
  },
  svg: {
    fontCache: 'global'
  }
};

// Load MathJax when the page is ready
document.addEventListener('DOMContentLoaded', function() {
  if (typeof MathJax !== 'undefined') {
    MathJax.typesetPromise();
  }
}); 