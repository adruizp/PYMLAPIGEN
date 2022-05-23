/**
 * Filtra una tabla HTML
 */
function filter() {
    var input, filter, column, index, table, tr, td, i, txtValue;

    //Obtener el input
    input = document.getElementById("filterInput");
    filter = input.value.toUpperCase();
    
    //Obtener la columna
    column = document.getElementById("filterColumn");
    index = column.selectedIndex + 1;

    
    //Procesar tabla
    table = document.getElementById("datasetTable");
    tr = table.getElementsByTagName("tr");
    for (i = 0; i < tr.length; i++) {
      td = tr[i].getElementsByTagName("td")[index];
      if (td) {
        txtValue = td.textContent || td.innerText;
        if (txtValue.toUpperCase().indexOf(filter) > -1) {
          tr[i].style.display = "";
        } else {
          tr[i].style.display = "none";
        }
      }       
    }
  }