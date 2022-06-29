/**
 * Deshabilita el atributo objetivo en experimentos de aprendizaje no supervisado
 */

var labelSelector = document.getElementById("inputLabel")
var labelTextIndicator = document.getElementById("clusteringSelected")
var select = document.getElementById("modelType")

labelTextIndicator.style.display = select.options[select.selectedIndex].parentElement.label == 'Clustering' ? '' : 'none'


function algorithmChange(){
   
    var problem = select.options[select.selectedIndex].parentElement.label
    
    isClustering = problem == 'Clustering'

    labelSelector.disabled = isClustering

    labelTextIndicator.style.display = isClustering ? '' : 'none'
}