function toggleTable() {
    var tableDiv = document.getElementById("cluster-table");
    var button = document.querySelector('.toggle-button');
    if (tableDiv.classList.contains('hidden')) {
        tableDiv.classList.remove('hidden');
        button.textContent = 'Ẩn Bảng Dữ Liệu';
    } else {
        tableDiv.classList.add('hidden');
        button.textContent = 'Hiện Bảng Dữ Liệu';
    }
}

function showRows() {
    var rowCount = document.getElementById('rowCount').value;
    var table = document.querySelector('#cluster-table table');
    var rows = table.querySelectorAll('tbody tr');

    if (rowCount === 'all') {
        rows.forEach(function(row) {
            row.style.display = '';
        });
        return;
    }

    rowCount = parseInt(rowCount, 10);
    rows.forEach(function(row, index) {
        if (index < rowCount) {
            row.style.display = '';
        } else {
            row.style.display = 'none';
        }
    });
}

window.onload = function() {
    showRows();
};
