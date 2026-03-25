let chart;
let featureChart;
let map;
let marker;


/* -------- LOCALITY COORDINATES -------- */

const localityCoords = {

"Dwarka Sector":[28.5921,77.0460],
"Rohini Sector":[28.7383,77.0822],
"Shahdara":[28.6735,77.2896],
"Vasant Kunj":[28.5245,77.1544],
"Paschim Vihar":[28.6692,77.1015],
"Alaknanda":[28.5317,77.2542],
"Vasundhara Enclave":[28.6065,77.3174],
"Punjabi Bagh":[28.6689,77.1259],
"Kalkaji":[28.5497,77.2588],
"Lajpat Nagar":[28.5677,77.2435],
"Other":[28.6139,77.2090]

};


/* -------- HEATMAP DATA -------- */

const heatData = [

[28.5921,77.0460,0.7],
[28.7383,77.0822,0.6],
[28.6735,77.2896,0.5],
[28.5245,77.1544,0.9],
[28.6692,77.1015,0.7],
[28.5317,77.2542,0.8],
[28.6065,77.3174,0.6],
[28.6689,77.1259,0.85],
[28.5497,77.2588,0.75],
[28.5677,77.2435,0.8]

];


/* -------- MAP INITIALIZATION -------- */

document.addEventListener("DOMContentLoaded", function(){

map = L.map('map').setView([28.6139,77.2090],11);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{
maxZoom:19
}).addTo(map);

/* Heatmap Layer */

L.heatLayer(heatData,{
radius:30,
blur:20,
maxZoom:13
}).addTo(map);

});


/* -------- PRICE PREDICTION -------- */
function animatePrice(finalPrice){

const priceElement = document.getElementById("price");

let start = 0;
let duration = 800;
let startTime = null;

function animate(timestamp){

if(!startTime) startTime = timestamp;

let progress = timestamp - startTime;
let value = Math.floor(progress / duration * finalPrice);

if(value > finalPrice){
value = finalPrice;
}

priceElement.innerText = "₹ " + value.toLocaleString();

if(progress < duration){
requestAnimationFrame(animate);
}

}

requestAnimationFrame(animate);

}
async function predictPrice(){

const form = document.getElementById("prediction-form");

const loader = document.getElementById("loader");
const priceElement = document.getElementById("price");

loader.style.display = "block";
priceElement.innerText = "Calculating...";

const formData = new FormData(form);

try{

const response = await fetch("/predict",{
method:"POST",
body:formData
});

const data = await response.json();

loader.style.display = "none";

if(data.price){

priceElement.innerText = "₹ " + data.price;

const area = parseInt(formData.get("Area"));
const predicted = parseInt(data.price.replace(/,/g,""));

updateChart(area,predicted);
updateMap(formData.get("Locality"));
updateFeatureChart();

/* AI Insight */

const insightText = document.getElementById("insightText");

const expected = area * parseInt(formData.get("Per_Sqft"));

if(predicted < expected*0.9){

insightText.innerText =
"💡 This property appears undervalued. It may be a good investment opportunity.";

}
else if(predicted > expected*1.1){

insightText.innerText =
"⚠ This property price is above expected market value. Consider negotiating.";

}
else{

insightText.innerText =
"✅ This property is priced close to the market average.";

}

}else{

animatePrice(predicted);

}

}catch(err){

console.error(err);

loader.style.display = "none";
priceElement.innerText = "Error displaying result";

}

}


/* -------- PRICE CHART -------- */

function updateChart(area,predicted){

const ctx = document.getElementById("priceChart").getContext("2d");

if(chart){
chart.destroy();
}

chart = new Chart(ctx,{
type:"line",
data:{
labels:[area-200, area, area+200],
datasets:[{
label:"Estimated Property Price",
data:[
predicted*0.8,
predicted,
predicted*1.2
],
borderColor:"#ff7a18",
backgroundColor:"rgba(255,122,24,0.2)",
tension:0.4
}]
},
options:{
responsive:true,
maintainAspectRatio:false,
plugins:{
legend:{
display:true
}
},
scales:{
y:{
beginAtZero:false
}
}
}
});

}


/* -------- MAP UPDATE -------- */

function updateMap(locality){

if(localityCoords[locality]){

const coords = localityCoords[locality];

map.setView(coords,13);

if(marker){
map.removeLayer(marker);
}

marker = L.marker(coords)
.addTo(map)
.bindPopup(locality)
.openPopup();

}

}


/* -------- FEATURE IMPORTANCE -------- */

function updateFeatureChart(){

const ctx = document.getElementById("featureChart").getContext("2d");

if(featureChart){
featureChart.destroy();
}

featureChart = new Chart(ctx,{
type:"bar",
data:{
labels:[
"Area",
"BHK",
"Bathroom",
"Locality",
"Parking",
"Furnishing"
],
datasets:[{
label:"Impact on Price",
data:[
0.9,
0.7,
0.6,
0.8,
0.4,
0.5
],
backgroundColor:"#ff7a18"
}]
},
options:{
responsive:true,
plugins:{
legend:{
display:false
}
},
scales:{
y:{
beginAtZero:true
}
}
}
});

}


/* -------- AUTO PREDICTION -------- */

const inputs =
document.querySelectorAll("#prediction-form input, #prediction-form select");

inputs.forEach(input => {

input.addEventListener("change", () => {

const area = document.querySelector('input[name="Area"]').value;
const sqft = document.querySelector('input[name="Per_Sqft"]').value;

if(area && sqft){
predictPrice();
}else{

document.getElementById("price").innerText = "₹ --";

}

});

});