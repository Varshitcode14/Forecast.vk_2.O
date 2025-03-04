document.addEventListener("DOMContentLoaded", () => {
    // Mobile navigation toggle
    const navToggle = document.querySelector(".nav-toggle")
    const nav = document.querySelector("nav")
  
    navToggle.addEventListener("click", () => {
      nav.classList.toggle("active")
      navToggle.setAttribute("aria-expanded", navToggle.getAttribute("aria-expanded") === "true" ? "false" : "true")
    })
  
    // Close mobile menu when clicking on a link
    const navLinks = document.querySelectorAll("nav ul li a")
    navLinks.forEach((link) => {
      link.addEventListener("click", () => {
        if (window.innerWidth <= 768) {
          nav.classList.remove("active")
          navToggle.setAttribute("aria-expanded", "false")
        }
      })
    })
  
    // Add active class to current page in navigation
    const currentLocation = window.location.pathname
    navLinks.forEach((link) => {
      const linkPath = link.getAttribute("href")
      if (
        currentLocation === linkPath ||
        (linkPath === "/" && (currentLocation === "/index.html" || currentLocation === "/"))
      ) {
        link.classList.add("active")
      }
    })
  
    // Add smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
      anchor.addEventListener("click", function (e) {
        e.preventDefault()
        document.querySelector(this.getAttribute("href")).scrollIntoView({
          behavior: "smooth",
        })
      })
    })
  
    // Add animation to cards when they come into view
    const animateOnScroll = () => {
      const cards = document.querySelectorAll(".card, .feature-card, .summary-card, .notification-item")
  
      cards.forEach((card) => {
        const cardPosition = card.getBoundingClientRect().top
        const screenPosition = window.innerHeight / 1.2
  
        if (cardPosition < screenPosition) {
          card.style.opacity = "1"
          card.style.transform = "translateY(0)"
        }
      })
    }
  
    // Set initial state for animation
    const elementsToAnimate = document.querySelectorAll(".card, .feature-card, .summary-card, .notification-item")
    elementsToAnimate.forEach((element) => {
      element.style.opacity = "0"
      element.style.transform = "translateY(20px)"
      element.style.transition = "opacity 0.5s ease, transform 0.5s ease"
    })
  
    // Run animation on load and scroll
    window.addEventListener("load", animateOnScroll)
    window.addEventListener("scroll", animateOnScroll)
  })
  
  