/**
 * Filtra el dataset del experimento
 */

var filterDiv = document.querySelector(".filter").cloneNode(true)
filterDiv.querySelector('.minus').style.display = ''


/**
 * Filtra una tabla HTML
 */
function filter() {
  var table, tr, td, i, j, tdValue, toDisplay;

  //Obtener los inputs
  var include = document.querySelectorAll(".filterInclude")
  var columns = document.querySelectorAll(".filterColumn")
  var ops = document.querySelectorAll(".filterOp")
  var inputs = document.querySelectorAll(".filterInput")

  //Tabla
  table = document.getElementById("datasetTable");
  //Filas
  tr = table.getElementsByTagName("tr");



  for (i = 0; i < tr.length; i++) {
    toDisplay = true;
    for (j = 0; j < inputs.length; j++) {

      //Solo aplicar filtros con valores escritos
      if (include[j].checked && inputs[j].value != "") {

        td = tr[i].getElementsByTagName("td")[columns[j].selectedIndex + 1];
        if (td && toDisplay) {
          tdValue = td.textContent || td.innerText;

          switch (ops[j].value) {
            case "contains":
              toDisplay = tdValue.toUpperCase().indexOf(inputs[j].value.toUpperCase()) > -1;
              break
            case "==":
              toDisplay = tdValue.trim() === inputs[j].value.trim();
              break
            case "<":
              toDisplay = +tdValue < +inputs[j].value;
              break
            case "<=":
              toDisplay = +tdValue <= +inputs[j].value;
              break
            case ">":
              toDisplay = +tdValue > +inputs[j].value;
              break
            case ">=":
              toDisplay = +tdValue >= +inputs[j].value;
              break
          }
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



function addfilter() {
  document.getElementById("aditionalFilters").appendChild(filterDiv.cloneNode(true));
}

function deletefilter(filterId) {
  filterId.parentElement.parentElement.remove()
  filter()
}