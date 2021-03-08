var items = Array("Is Python better then PHP for web development?",
    "What is better for deep learning Python or Matlab?",
    "Is Python better than matlab for web development?", "What is better tea or coffee?");

var search_items = Array("single gender schools", "death penalty", "asylum", "chinese medicine");

var item = items[Math.floor(Math.random() * items.length)];

const displacy = new displaCyENT('https://api.explosion.ai/displacy/ent/', {
                container: '#displacy1'
            });



$('#labelTextt').val(item)
document.getElementById("button_label1").disabled = false;
document.getElementById("button_label1").addEventListener('click',function ()
    {
    document.getElementById("Outputt").innerHTML = "... Please wait ...";
    document.getElementById("button_label1").innerHTML = "Processing";
    document.getElementById("button_label1").disabled = true;
    $("#button_label1").prop('disabled', true);
    document.getElementById("displacy1").innerHTML = "Wait";
    $('#Outputt').val(" ")
    $.post("./label_text", {
        username: document.getElementById("labelTextt").value,
        classifier: document.getElementById("model").value,
        extractor: document.getElementById("extractor").value,
    })
        .done(function (data) {
            document.getElementById("button_label1").disabled = false;
        document.getElementById("displacy1").innerHTML = "Searching ...";
            document.getElementById("button_label1").innerHTML = "Answer";
            $('#Outputt').val(data["full_answer"])
            document.getElementById("displacy1").innerHTML = "Extracting ..."
        
            $('#displacy-container1').show();
            $("#displacy1").empty();
            var d = data["spans"]
            console.log("d")
            console.log(d)
            $("#displacy1").val(data["spans"])
            const text = document.getElementById("labelTextt").value;
            const spans = data["spans"];
            const ents = ['obj', 'pred'];
            displacy.render(text, spans, ents);
            

        })
        .fail(function (jqxhr, textStatus, error) {
            $('#Outputt').val("Something went wrong")
            var err = textStatus + ", " + error;
            $('#Outputt').val(err)
            console.log("Request Failed: " + err);
            document.getElementById("button_label1").disabled = false;
            document.getElementById("button_label1").innerHTML = "Answer0";
        });
    }  ); 

