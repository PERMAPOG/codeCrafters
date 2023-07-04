$(document).ready(function() {
    const videoElement = document.getElementById('videoElement');
    const canvas = document.getElementById('canvas');
    const context = canvas.getContext('2d');

    // Retrieve CSRF token from Django
    const csrftoken = document.querySelector('[name=csrfmiddlewaretoken]').value;

    $('#captureButton').click(function() {
        context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(function(blob) {
            let formData = new FormData();

            formData.append('image', blob, 'image.jpg');
            formData.append('csrfmiddlewaretoken', csrftoken);

            $.ajax({
                url: '/face_mesh/',
                method: 'POST',
                processData: false,  // tell jQuery not to process the data
                contentType: false,  // tell jQuery not to set contentType
                data: formData,
                success: function(response) {
                    console.log('Image sent successfully');
                },
                error: function(error) {
                    console.log('Error sending image');
                }
            });
        }, 'image/jpeg');
    });

    // Access the camera feed
    $('#startButton').click(function() {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(function(stream) {
                videoElement.srcObject = stream;
            })
            .catch(function(err) {
                console.log("An error occurred: " + err);
            });
    });
});
