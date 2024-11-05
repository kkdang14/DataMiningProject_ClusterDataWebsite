// Hàm để ẩn/hiện bảng và thay đổi nội dung nút
function toggleTable(tableId, button) {
    const tableDiv = document.getElementById(tableId);
    if (tableDiv.style.display === 'none' || !tableDiv.style.display) {
        tableDiv.style.display = 'block';
        button.textContent = 'Ẩn Bảng Dữ Liệu';
    } else {
        tableDiv.style.display = 'none';
        button.textContent = 'Hiện Bảng Dữ Liệu';
    }
}

// Hàm để hiển thị số lượng dòng dựa trên lựa chọn
function showRows(tableId) {
    const rowCount = document.getElementById(tableId === 'cluster-table' ? 'rowCount' : 'stepRowCount').value;
    const table = document.querySelector(`#${tableId} table`);
    const rows = table.querySelectorAll('tbody tr');

    rows.forEach((row, index) => {
        row.style.display = (rowCount === 'all' || index < rowCount) ? '' : 'none';
    });
}

// Khởi tạo hiển thị mặc định
window.onload = function() {
    showRows('cluster-table');
    showRows('step-table');
};
