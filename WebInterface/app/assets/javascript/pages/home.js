document.addEventListener("DOMContentLoaded", function () {
    const checkboxes = document.querySelectorAll('input[type=checkbox]');
    const dataSelector = document.querySelector('#data_option');
    const confirmButton = document.querySelector('#submit-button');


    function enableApplyButton() {
        let enable = false;
        checkboxes.forEach(checkbox => {
            if (checkbox.checked) {
                enable = true;
            }
        });
        if (dataSelector.value && enable) {
            confirmButton.disabled = false;
        } else {
            confirmButton.disabled = true;
        }
    }

    checkboxes.forEach(checkbox => {
        checkbox.addEventListener('change', enableApplyButton);
    });

    dataSelector.addEventListener('change', enableApplyButton);
});
