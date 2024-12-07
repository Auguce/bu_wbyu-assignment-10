document.getElementById("query_type").addEventListener("change", function (event) {
    const queryType = event.target.value;

    const textQueryGroup = document.getElementById("text-query-group");
    const imageQueryGroup = document.getElementById("image-query-group");
    const weightGroup = document.getElementById("weight-group");
    const pcaGroup = document.getElementById("pca-group");
    const pcaKGroup = document.getElementById("pca-k-group");

    if (queryType === "text") {
        textQueryGroup.style.display = "block";
        imageQueryGroup.style.display = "none";
        pcaGroup.style.display = "none";
        pcaKGroup.style.display = "none";
        weightGroup.style.display = "none";
    } else if (queryType === "image") {
        textQueryGroup.style.display = "none";
        imageQueryGroup.style.display = "block";
        pcaGroup.style.display = "block";
        weightGroup.style.display = "none";
        togglePcaKInput();
    } else if (queryType === "hybrid") {
        textQueryGroup.style.display = "block";
        imageQueryGroup.style.display = "block";
        pcaGroup.style.display = "block";
        weightGroup.style.display = "block";
        togglePcaKInput();
    }
});

document.getElementById("use_pca").addEventListener("change", togglePcaKInput);

function togglePcaKInput() {
    const usePcaCheckbox = document.getElementById("use_pca");
    const pcaKGroup = document.getElementById("pca-k-group");
    if (usePcaCheckbox.checked) {
        pcaKGroup.style.display = "block";
    } else {
        pcaKGroup.style.display = "none";
    }
}

// Initialize display settings
document.getElementById("query_type").dispatchEvent(new Event('change'));

document.getElementById("search-form").addEventListener("submit", function (event) {
    event.preventDefault();

    const form = document.getElementById("search-form");
    const formData = new FormData(form);

    fetch("/search_images", {
        method: "POST",
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            const resultsDiv = document.getElementById("results");
            const container = document.getElementById("results-container");
            container.innerHTML = "";

            if (data.results) {
                resultsDiv.style.display = "block";
                data.results.forEach(res => {
                    const imgDiv = document.createElement("div");
                    imgDiv.classList.add("result-item");

                    const imgElem = document.createElement("img");
                    imgElem.src = `/coco_images_resized/${res.image_name}`;
                    imgElem.alt = res.image_name;

                    const info = document.createElement("p");
                    info.textContent = `Similarity: ${(res.similarity * 100).toFixed(2)}%`;

                    imgDiv.appendChild(imgElem);
                    imgDiv.appendChild(info);
                    container.appendChild(imgDiv);
                });
            } else if (data.error) {
                alert(data.error);
            }
        })
        .catch(error => {
            console.error("Error searching images:", error);
            alert("An error occurred while searching images.");
        });
});
