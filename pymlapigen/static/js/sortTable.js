/**
 * Ordena una tabla HTML
 * 
 * @param {HTMLTableElement} table Tabla a ordenar 
 * @param {number} column Indice de la tabla a ordenar
 * @param {boolean} asc Determina si se ordenara de forma ascendente 
 */
function sortTableByColumn(table, column, asc = true){
    const dirModifier = asc ? 1 : -1;
    const tBody = table.tBodies[0];
    const rows = Array.from(tBody.querySelectorAll("tr"));

    // Ordenar cada columna
    const sortedRows = rows.sort((a, b) =>{
        const aColText = a.querySelector(`td:nth-child(${ column + 1 })`).textContent.trim();
        const bColText = b.querySelector(`td:nth-child(${ column + 1 })`).textContent.trim();
        
        var sortValue;

        //Comprueba si son columnas de strings.
        if (isNaN(aColText) || isNaN(bColText)){
            sortValue = aColText > bColText ? (1 * dirModifier) : (-1 * dirModifier);
        }

        //Columnas numéricas
        else{
            sortValue = +aColText > +bColText ? (1 * dirModifier) : (-1 * dirModifier);
        }

        return sortValue
    });

    // Borra todos los TRs de la tabla
    while (tBody.firstChild){
        tBody.removeChild(tBody.firstChild);
    }

    // Añade las filas ordenadas
    tBody.append(...sortedRows);

    // Recordar que tabla ha sido ordendada
    table.querySelectorAll("th").forEach(th => th.classList.remove("th-sort-asc", "th-sort-desc"));
    table.querySelector(`th:nth-child(${ column + 1})`).classList.toggle("th-sort-asc", asc);
    table.querySelector(`th:nth-child(${ column + 1})`).classList.toggle("th-sort-desc", !asc);

}


document.querySelectorAll(".table-sortable th").forEach(headerCell => {
    headerCell.addEventListener("click", () => {
        console.log(headerCell)

        const tableElement = headerCell.parentElement.parentElement.parentElement;
        const headerIndex = Array.prototype.indexOf.call(headerCell.parentElement.children, headerCell);
        const currentIsAscending = headerCell.classList.contains("th-sort-asc");

        sortTableByColumn(tableElement, headerIndex, !currentIsAscending);
    });
});
