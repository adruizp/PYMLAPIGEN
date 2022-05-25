var filterDiv = document.querySelector(".filter").cloneNode(true)
filterDiv.querySelector('.minus').style.display = ''


/**
 * Filtra una tabla HTML
 */
function filter() {
  var filter, column, index, table, tr, td, i, j, txtValue, toDisplay;

  //Obtener los inputs
  var columns = document.querySelectorAll(".filterColumn")
  var inputs = document.querySelectorAll(".filterInput")

  for (i = 0; i < inputs.length; i++) {
    console.log(i)

    filter = inputs[i].value.toUpperCase();

    //Obtener la columna
    column = columns[i];
    index = column.selectedIndex + 1;

    //Procesar tabla
    table = document.getElementById("datasetTable");
    tr = table.getElementsByTagName("tr");




    for (i = 0; i < tr.length; i++) {
      toDisplay = true;
      for (j = 0; j < inputs.length; j++) {
        td = tr[i].getElementsByTagName("td")[columns[j].selectedIndex + 1];
        if (td && toDisplay) {
          txtValue = td.textContent || td.innerText;
          if (txtValue.toUpperCase().indexOf(inputs[j].value.toUpperCase()) < 0) {
            toDisplay = false;
            console.log("NO ES CORRECTOP")
          }
        }
      }
      if (toDisplay) {
        tr[i].style.display = "";
      } else {
        tr[i].style.display = "none";
      }
    }
  }


}

function addfilter() {
  document.getElementById("aditionalFilters").appendChild(filterDiv.cloneNode(true));
}

function deletefilter(filterId) {
  console.log("WORKS")
  filterId.parentElement.parentElement.remove()

}