'use strict';
// No analytics. All navigation and figures work without third-party scripts.
const contents = document.querySelector('.contents');
const toggle = document.querySelector('.contents-toggle');
toggle?.addEventListener('click', () => {
  const open = contents.classList.toggle('expanded');
  toggle.setAttribute('aria-expanded', String(open));
  toggle.querySelector('span').textContent = open ? '−' : '+';
});
const links = [...document.querySelectorAll('.contents nav a')];
links.forEach(link => link.addEventListener('click', () => {
  contents.classList.remove('expanded');
  toggle.setAttribute('aria-expanded', 'false');
  toggle.querySelector('span').textContent = '+';
}));
if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver(entries => {
    const visible = entries.filter(e => e.isIntersecting).sort((a,b) => a.boundingClientRect.top - b.boundingClientRect.top);
    if (!visible.length) return;
    links.forEach(link => {
      const active = link.hash === '#' + visible[0].target.id;
      link.classList.toggle('active', active);
      if (active) link.setAttribute('aria-current','location'); else link.removeAttribute('aria-current');
    });
  }, {rootMargin: '-10% 0px -65% 0px', threshold: 0});
  document.querySelectorAll('.report-section').forEach(section => observer.observe(section));
}
document.querySelector('.video-load')?.addEventListener('click', () => {
  const frame = document.createElement('iframe');
  frame.src = 'https://www.youtube-nocookie.com/embed/mWIeYK-iH-M?autoplay=1';
  frame.title = 'TurtleBot3 final demonstration: three navigation conditions';
  frame.allow = 'accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share';
  frame.allowFullscreen = true;
  document.querySelector('#demo').replaceChildren(frame);
});
// Expand technical details when printing, then restore the reader's choices.
let printOpened = [];
window.addEventListener('beforeprint', () => {
  printOpened = [...document.querySelectorAll('details:not([open])')];
  printOpened.forEach(el => el.open = true);
});
window.addEventListener('afterprint', () => printOpened.forEach(el => el.open = false));
