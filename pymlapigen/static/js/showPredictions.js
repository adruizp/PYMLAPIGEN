var table = document.getElementById("predictionTable")
table.style.display = 'none'

function toggle(){
    var div = document.querySelector(".showable")
    var isHidden = table.style.display == 'none'
    
    if (!isHidden){
        table.style.display = 'none'
    }
    else table.style.display = ''

    div.classList.toggle("showable-enabled", isHidden);
    div.classList.toggle("showable-disabled", !isHidden);
}