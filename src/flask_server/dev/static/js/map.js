var map = L.map('map').setView([0, 0], 2);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
}).addTo(map);

fetch('/geojson')
    .then(response => response.json())
    .then(data => {
        L.geoJSON(data).addTo(map);
    });